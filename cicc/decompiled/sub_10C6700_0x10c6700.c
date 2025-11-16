// Function: sub_10C6700
// Address: 0x10c6700
//
__int64 __fastcall sub_10C6700(__int64 a1, __int64 a2, char a3, unsigned int **a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v9; // rdx
  __int64 v11; // rsi
  _BYTE *v12; // r14
  __int64 v14; // rax
  int v15; // r14d
  __int64 v16; // rdx
  bool v17; // al
  int v18; // eax
  __int64 v19; // r15
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  unsigned __int8 *v25; // rdx
  unsigned int v26; // ecx
  unsigned __int8 *v27; // rax
  unsigned int v28; // ecx
  int v29; // eax
  int v30; // [rsp+8h] [rbp-88h]
  bool v31; // [rsp+Ch] [rbp-84h]
  unsigned int v32; // [rsp+Ch] [rbp-84h]
  int v33; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-80h]
  unsigned int v35; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  int v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  int v40; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v41; // [rsp+18h] [rbp-78h]
  unsigned int v42; // [rsp+18h] [rbp-78h]
  _BYTE v44[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v45; // [rsp+50h] [rbp-40h]

  if ( !a1 )
    return 0;
  v5 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v5 != 85 )
    return 0;
  v9 = *(_QWORD *)(v5 - 32);
  if ( !v9 )
    return 0;
  if ( *(_BYTE *)v9 )
    return 0;
  v11 = *(_QWORD *)(v5 + 80);
  if ( *(_QWORD *)(v9 + 24) != v11 )
    return 0;
  if ( *(_DWORD *)(v9 + 36) != 66 )
    return 0;
  v37 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  if ( !v37 )
    return 0;
  v12 = *(_BYTE **)(a1 - 32);
  if ( !v12 )
    BUG();
  if ( *v12 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v12 + 1) + 8LL) - 17;
    if ( (unsigned int)v22 > 1 )
      return 0;
    if ( *v12 > 0x15u )
      return 0;
    v11 = 0;
    v23 = sub_AD7630(*(_QWORD *)(a1 - 32), 0, v22);
    v12 = v23;
    if ( !v23 || *v23 != 17 )
      return 0;
  }
  if ( *((_DWORD *)v12 + 8) > 0x40u )
  {
    v33 = *((_DWORD *)v12 + 8);
    if ( v33 - (unsigned int)sub_C444A0((__int64)(v12 + 24)) > 0x40 )
      return 0;
    v14 = **((_QWORD **)v12 + 3);
  }
  else
  {
    v14 = *((_QWORD *)v12 + 3);
  }
  if ( v14 != 1 )
    return 0;
  v15 = sub_B53900(a1);
  if ( !a2 || v37 != *(_QWORD *)(a2 - 64) )
    return 0;
  v16 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v16 == 17 )
  {
    if ( *(_DWORD *)(v16 + 32) <= 0x40u )
    {
      v17 = *(_QWORD *)(v16 + 24) == 0;
    }
    else
    {
      v38 = *(_DWORD *)(v16 + 32);
      v17 = v38 == (unsigned int)sub_C444A0(v16 + 24);
    }
  }
  else
  {
    v39 = *(_QWORD *)(v16 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 > 1 || *(_BYTE *)v16 > 0x15u )
      return 0;
    v11 = 0;
    v34 = *(unsigned __int8 **)(a2 - 32);
    v24 = sub_AD7630(v16, 0, v16);
    v25 = v34;
    if ( !v24 || *v24 != 17 )
    {
      if ( *(_BYTE *)(v39 + 8) == 17 )
      {
        v30 = *(_DWORD *)(v39 + 32);
        if ( v30 )
        {
          v31 = 0;
          v26 = 0;
          while ( 1 )
          {
            v35 = v26;
            v41 = v25;
            v27 = (unsigned __int8 *)sub_AD69F0(v25, v26);
            if ( !v27 )
              break;
            v11 = *v27;
            v25 = v41;
            v28 = v35;
            if ( (_BYTE)v11 != 13 )
            {
              if ( (_BYTE)v11 != 17 )
                break;
              v11 = *((unsigned int *)v27 + 8);
              if ( (unsigned int)v11 <= 0x40 )
              {
                v31 = *((_QWORD *)v27 + 3) == 0;
              }
              else
              {
                v32 = *((_DWORD *)v27 + 8);
                v36 = v41;
                v42 = v28;
                v29 = sub_C444A0((__int64)(v27 + 24));
                v11 = v32;
                v25 = v36;
                v28 = v42;
                v31 = v32 == v29;
              }
              if ( !v31 )
                break;
            }
            v26 = v28 + 1;
            if ( v30 == v26 )
            {
              if ( v31 )
                goto LABEL_21;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( *((_DWORD *)v24 + 8) <= 0x40u )
    {
      v17 = *((_QWORD *)v24 + 3) == 0;
    }
    else
    {
      v40 = *((_DWORD *)v24 + 8);
      v17 = v40 == (unsigned int)sub_C444A0((__int64)(v24 + 24));
    }
  }
  if ( !v17 )
    return 0;
LABEL_21:
  v18 = sub_B53900(a2);
  v19 = *(_QWORD *)(a1 - 64);
  if ( a3 )
  {
    v6 = 0;
    if ( v15 == 33 && v18 == 33 )
    {
      sub_B44F30((unsigned __int8 *)v19);
      sub_B44B50((__int64 *)v19, v11);
      sub_B44A60(v19);
      sub_F15FC0(*(_QWORD *)(a5 + 40), v19);
      v45 = 257;
      v20 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v19 + 8), 1, 0);
      return sub_92B530(a4, 0x22u, v19, v20, (__int64)v44);
    }
  }
  else
  {
    v6 = 0;
    if ( v15 == 32 && v18 == 32 )
    {
      sub_B44F30((unsigned __int8 *)v19);
      sub_B44B50((__int64 *)v19, v11);
      sub_B44A60(v19);
      sub_F15FC0(*(_QWORD *)(a5 + 40), v19);
      v45 = 257;
      v21 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v19 + 8), 2, 0);
      return sub_92B530(a4, 0x24u, v19, v21, (__int64)v44);
    }
  }
  return v6;
}
