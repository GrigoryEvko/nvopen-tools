// Function: sub_E331B0
// Address: 0xe331b0
//
__int64 __fastcall sub_E331B0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // r12d
  unsigned __int64 v5; // r15
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // r9
  unsigned __int64 v10; // r8
  size_t v12; // rdi
  const void *v13; // rdx
  char v14; // r8
  char v15; // r13
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rax
  unsigned int v20; // eax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rdi
  char v24; // si
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // r13
  size_t v27; // r14
  char v28; // [rsp+Bh] [rbp-55h]
  char v29; // [rsp+Ch] [rbp-54h]
  size_t v30; // [rsp+10h] [rbp-50h] BYREF
  const void *v31; // [rsp+18h] [rbp-48h]
  char v32; // [rsp+20h] [rbp-40h]

  v4 = *(unsigned __int8 *)(a1 + 49);
  if ( (_BYTE)v4 || (v5 = *(_QWORD *)(a1 + 8), v5 >= *(_QWORD *)a1) )
  {
    *(_BYTE *)(a1 + 49) = 1;
    return 0;
  }
  *(_QWORD *)(a1 + 8) = v5 + 1;
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(a1 + 24);
  if ( v7 >= v8 )
  {
LABEL_11:
    *(_BYTE *)(a1 + 49) = 1;
    goto LABEL_12;
  }
  v9 = *(_QWORD *)(a1 + 32);
  v10 = v7 + 1;
  *(_QWORD *)(a1 + 40) = v7 + 1;
  switch ( *(_BYTE *)(v9 + v7) )
  {
    case 'B':
      if ( v8 > v10 && *(_BYTE *)(v9 + v7 + 1) == 95 )
      {
        v18 = v7 + 2;
        v19 = 0;
        *(_QWORD *)(a1 + 40) = v18;
      }
      else
      {
        v19 = sub_E31BC0(a1);
        a2 = (unsigned int)a2;
        if ( *(_BYTE *)(a1 + 49) )
          goto LABEL_11;
        v18 = *(_QWORD *)(a1 + 40);
      }
      if ( v18 <= v19 )
        goto LABEL_11;
      v4 = *(unsigned __int8 *)(a1 + 48);
      if ( (_BYTE)v4 )
      {
        *(_QWORD *)(a1 + 40) = v19;
        v20 = sub_E331B0(a1, a2, a3);
        *(_QWORD *)(a1 + 40) = v18;
        v4 = v20;
      }
      goto LABEL_12;
    case 'C':
      sub_E31E90(a1, 115);
      sub_E31F10((__int64)&v30, a1);
      v12 = v30;
      v13 = v31;
      if ( *(_BYTE *)(a1 + 49) || !*(_BYTE *)(a1 + 48) )
        goto LABEL_12;
      if ( !v32 )
        goto LABEL_63;
      if ( !(unsigned __int8)sub_E31600(v30, (__int64)v31, (__int64 *)(a1 + 56)) )
        goto LABEL_11;
      goto LABEL_12;
    case 'I':
      sub_E331B0(a1, a2, 0);
      if ( !(_DWORD)a2 )
        sub_E31C60(a1, 2u, "::");
      v16 = 0;
      sub_E31C60(a1, 1u, "<");
      if ( *(_BYTE *)(a1 + 49) )
        goto LABEL_30;
      break;
    case 'M':
      sub_E33710(a1);
      sub_E31C60(a1, 1u, "<");
      sub_E327B0(a1);
      sub_E31C60(a1, 1u, ">");
      goto LABEL_12;
    case 'N':
      if ( v8 <= v10 )
        goto LABEL_11;
      *(_QWORD *)(a1 + 40) = v7 + 2;
      v14 = *(_BYTE *)(v9 + v7 + 1);
      if ( (unsigned __int8)(v14 - 97) > 0x19u )
      {
        if ( (unsigned __int8)(v14 - 65) > 0x19u )
          goto LABEL_11;
        v29 = *(_BYTE *)(v9 + v7 + 1);
        sub_E331B0(a1, a2, 0);
        v26 = sub_E31E90(a1, 115);
        sub_E31F10((__int64)&v30, a1);
        v27 = v30;
        v28 = v32;
        sub_E31C60(a1, 3u, "::{");
        if ( v29 == 67 )
        {
          sub_E31C60(a1, 7u, "closure");
        }
        else if ( v29 == 83 )
        {
          sub_E31C60(a1, 4u, "shim");
        }
        else
        {
          sub_E31570(a1, v29);
        }
        if ( v27 )
        {
          sub_E31C60(a1, 1u, ":");
          if ( !*(_BYTE *)(a1 + 49) )
          {
            if ( *(_BYTE *)(a1 + 48) )
            {
              if ( v28 )
              {
                if ( !(unsigned __int8)sub_E31600(v30, (__int64)v31, (__int64 *)(a1 + 56)) )
                  *(_BYTE *)(a1 + 49) = 1;
              }
              else
              {
                sub_E31C60(a1, v30, v31);
              }
            }
          }
        }
        sub_E31570(a1, 35);
        sub_E31D10(a1, v26);
        sub_E31570(a1, 125);
        goto LABEL_12;
      }
      sub_E331B0(a1, a2, 0);
      sub_E31E90(a1, 115);
      sub_E31F10((__int64)&v30, a1);
      v15 = v32;
      if ( !v30 )
        goto LABEL_12;
      sub_E31C60(a1, 2u, "::");
      v12 = v30;
      if ( *(_BYTE *)(a1 + 49) || !*(_BYTE *)(a1 + 48) )
        goto LABEL_12;
      if ( !v15 )
      {
        v13 = v31;
LABEL_63:
        sub_E31C60(a1, v12, v13);
        goto LABEL_12;
      }
      if ( (unsigned __int8)sub_E31600(v30, (__int64)v31, (__int64 *)(a1 + 56)) )
        goto LABEL_12;
      goto LABEL_11;
    case 'X':
      sub_E33710(a1);
      goto LABEL_14;
    case 'Y':
LABEL_14:
      sub_E31C60(a1, 1u, "<");
      sub_E327B0(a1);
      sub_E31C60(a1, 4u, " as ");
      sub_E331B0(a1, 1, 0);
      sub_E31C60(a1, 1u, ">");
      goto LABEL_12;
    default:
      goto LABEL_11;
  }
  while ( 1 )
  {
    v17 = *(_QWORD *)(a1 + 40);
    if ( v17 < *(_QWORD *)(a1 + 24) && *(_BYTE *)(*(_QWORD *)(a1 + 32) + v17) == 69 )
      break;
    if ( v16 )
      sub_E31C60(a1, 2u, ", ");
    if ( *(_BYTE *)(a1 + 49) )
      goto LABEL_44;
    v21 = *(_QWORD *)(a1 + 40);
    v22 = *(_QWORD *)(a1 + 24);
    if ( v21 >= v22 )
      goto LABEL_44;
    v23 = *(_QWORD *)(a1 + 32);
    v24 = *(_BYTE *)(v23 + v21);
    if ( v24 == 76 )
    {
      *(_QWORD *)(a1 + 40) = v21 + 1;
      if ( v22 > v21 + 1 && *(_BYTE *)(v23 + v21 + 1) == 95 )
      {
        v25 = 0;
        *(_QWORD *)(a1 + 40) = v21 + 2;
      }
      else
      {
        v25 = sub_E31BC0(a1);
      }
      sub_E31E00(a1, v25);
      goto LABEL_45;
    }
    if ( v24 == 75 )
    {
      *(_QWORD *)(a1 + 40) = v21 + 1;
      sub_E323E0(a1);
    }
    else
    {
LABEL_44:
      sub_E327B0(a1);
    }
LABEL_45:
    ++v16;
    if ( *(_BYTE *)(a1 + 49) )
      goto LABEL_30;
  }
  *(_QWORD *)(a1 + 40) = v17 + 1;
LABEL_30:
  if ( a3 == 1 )
    v4 = 1;
  else
    sub_E31C60(a1, 1u, ">");
LABEL_12:
  *(_QWORD *)(a1 + 8) = v5;
  return v4;
}
