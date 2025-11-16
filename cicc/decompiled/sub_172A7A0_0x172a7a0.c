// Function: sub_172A7A0
// Address: 0x172a7a0
//
_QWORD *__fastcall sub_172A7A0(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rbx
  char v9; // al
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // r15
  int v14; // eax
  int v15; // eax
  __int64 **v16; // rax
  __int64 v17; // r13
  int v18; // eax
  unsigned int v19; // r14d
  __int64 **v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rdi
  unsigned __int8 *v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // rax
  unsigned int v28; // eax
  _QWORD *v29; // rdx
  unsigned int v30; // edx
  __int64 v31; // rax
  char v32; // cl
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // [rsp+0h] [rbp-90h]
  int v35; // [rsp+Ch] [rbp-84h]
  __int64 **v36; // [rsp+10h] [rbp-80h]
  unsigned int v37; // [rsp+10h] [rbp-80h]
  __int64 *v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h] BYREF
  __int16 v40; // [rsp+30h] [rbp-60h]
  __int64 v41[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v42; // [rsp+50h] [rbp-40h]

  v5 = *(a2 - 6);
  v6 = *(_QWORD *)(v5 + 8);
  if ( !v6 )
    return 0;
  v7 = *(_QWORD **)(v6 + 8);
  if ( v7 )
    return 0;
  v9 = *(_BYTE *)(v5 + 16);
  v11 = *(a2 - 3);
  if ( v9 == 35 )
    goto LABEL_16;
  if ( v9 != 5 )
  {
    if ( v9 != 39 && v9 != 48 && v9 != 47 )
    {
      if ( v9 == 37 )
      {
        v13 = *(_QWORD *)(v5 - 48);
        if ( *(_BYTE *)(v13 + 16) <= 0x10u && v11 == *(_QWORD *)(v5 - 24) )
          goto LABEL_18;
      }
      return 0;
    }
LABEL_16:
    if ( v11 == *(_QWORD *)(v5 - 48) )
    {
      v13 = *(_QWORD *)(v5 - 24);
      if ( *(_BYTE *)(v13 + 16) <= 0x10u )
        goto LABEL_18;
    }
    return 0;
  }
  if ( (*(_WORD *)(v5 + 18) != 11
     || v11 != *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))
     || (v13 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)))) == 0)
    && (*(_WORD *)(v5 + 18) != 15
     || v11 != *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))
     || (v13 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)))) == 0)
    && (*(_WORD *)(v5 + 18) != 24
     || v11 != *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))
     || (v13 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)))) == 0)
    && (*(_WORD *)(v5 + 18) != 23
     || v11 != *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))
     || (v13 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)))) == 0) )
  {
    if ( *(_WORD *)(v5 + 18) != 13 )
      return 0;
    v12 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
    v13 = *(_QWORD *)(v5 - 24 * v12);
    if ( !v13 || v11 != *(_QWORD *)(v5 + 24 * (1 - v12)) )
      return 0;
  }
LABEL_18:
  v14 = *(unsigned __int8 *)(v11 + 16);
  if ( (unsigned __int8)v14 > 0x17u )
  {
    v15 = v14 - 24;
  }
  else
  {
    if ( (_BYTE)v14 != 5 )
      return v7;
    v15 = *(unsigned __int16 *)(v11 + 18);
  }
  if ( v15 == 37 )
  {
    v16 = (*(_BYTE *)(v11 + 23) & 0x40) != 0
        ? *(__int64 ***)(v11 - 8)
        : (__int64 **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
    v38 = *v16;
    if ( *v16 )
    {
      if ( !(unsigned __int8)sub_1648D00(v11, 3) )
      {
        v17 = *a2;
        if ( *(_BYTE *)(*a2 + 8) == 16 || (unsigned __int8)sub_1705440(a1, v17, *v38) )
        {
          v18 = *(unsigned __int8 *)(v5 + 16);
          v19 = v18 - 24;
          v20 = (__int64 **)*v38;
          if ( (unsigned int)(v18 - 47) > 1 )
          {
LABEL_29:
            v21 = sub_15A43B0(v13, v20, 0);
            v42 = 257;
            v22 = *(_QWORD *)(a1 + 8);
            if ( v19 == 13 )
              v23 = (unsigned __int8 *)sub_17066B0(v22, 0xDu, v21, (__int64)v38, v41, 0, a3, a4, a5);
            else
              v23 = (unsigned __int8 *)sub_17066B0(v22, v19, (__int64)v38, v21, v41, 0, a3, a4, a5);
            v24 = *(_QWORD *)(a1 + 8);
            v40 = 257;
            v25 = sub_1729500(v24, v23, (__int64)v38, &v39, a3, a4, a5);
            v42 = 257;
            v26 = (__int64)v25;
            v27 = sub_1648A60(56, 1u);
            v7 = v27;
            if ( v27 )
              sub_15FC690((__int64)v27, v26, v17, (__int64)v41, 0);
            return v7;
          }
          v36 = (__int64 **)*v38;
          v28 = sub_16431D0(*v38);
          v20 = v36;
          if ( *(_BYTE *)(v13 + 16) == 13 )
          {
            v29 = *(_QWORD **)(v13 + 24);
            if ( *(_DWORD *)(v13 + 32) > 0x40u )
              v29 = (_QWORD *)*v29;
            if ( v28 > (unsigned __int64)v29 )
              goto LABEL_29;
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16 )
              return v7;
            v35 = *(_QWORD *)(*(_QWORD *)v13 + 32LL);
            if ( !v35 )
              goto LABEL_29;
            v30 = 0;
            v34 = v28;
            while ( 1 )
            {
              v37 = v30;
              v31 = sub_15A0A60(v13, v30);
              if ( !v31 )
                break;
              v32 = *(_BYTE *)(v31 + 16);
              if ( v32 != 9 )
              {
                if ( v32 != 13 )
                  break;
                v33 = *(_DWORD *)(v31 + 32) <= 0x40u ? *(_QWORD *)(v31 + 24) : **(_QWORD **)(v31 + 24);
                if ( v34 <= v33 )
                  break;
              }
              v30 = v37 + 1;
              if ( v35 == v37 + 1 )
              {
                v20 = (__int64 **)*v38;
                goto LABEL_29;
              }
            }
          }
        }
      }
    }
  }
  return v7;
}
