// Function: sub_F947D0
// Address: 0xf947d0
//
__int64 __fastcall sub_F947D0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        __int64 a9,
        int a10)
{
  __int64 v10; // r12
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v19; // rax
  char v20; // r11
  int v21; // edx
  int v22; // ecx
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // r9
  _QWORD *v27; // r12
  char v28; // al
  char v29; // [rsp+Bh] [rbp-45h]
  char v30; // [rsp+10h] [rbp-40h]
  _QWORD *v31; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+18h] [rbp-38h]
  unsigned __int8 v34; // [rsp+18h] [rbp-38h]

  if ( (_DWORD)qword_4F8C9A8 == a10 )
    return 0;
  v10 = a1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v12 = *(_QWORD *)(a1 + 40);
    if ( v12 == a2 )
      return 0;
    v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v13 == v12 + 48 )
      goto LABEL_39;
    if ( !v13 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_39:
      BUG();
    if ( *(_BYTE *)(v13 - 24) == 31 )
    {
      v14 = *(_DWORD *)(v13 - 20) & 0x7FFFFFF;
      if ( (_DWORD)v14 != 3 && a2 == *(_QWORD *)(v13 - 56) && !(unsigned __int8)sub_B19060(a4, a1, v14, a4) )
      {
        v30 = sub_991A70((unsigned __int8 *)a1, a3, a9, 0, 0, 1u, 0);
        if ( !v30 )
          return 0;
        v19 = sub_F946A0(a8, a1, 3u);
        v20 = v30;
        v22 = v21;
        if ( v21 == 1 )
          *(_DWORD *)(a5 + 8) = 1;
        else
          v22 = *(_DWORD *)(a5 + 8);
        if ( __OFADD__(*(_QWORD *)a5, v19) )
        {
          if ( v19 <= 0 )
          {
            *(_QWORD *)a5 = 0x8000000000000000LL;
            if ( a7 == v22 )
              goto LABEL_24;
            goto LABEL_33;
          }
          v23 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v23 = *(_QWORD *)a5 + v19;
        }
        *(_QWORD *)a5 = v23;
        if ( a7 == v22 )
        {
          if ( v23 <= a6 )
            goto LABEL_24;
LABEL_21:
          if ( (_BYTE)qword_4F8CA88 && *(_DWORD *)(a4 + 20) == *(_DWORD *)(a4 + 24) && !(a10 | v22) )
          {
LABEL_24:
            v24 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
            {
              v25 = *(_QWORD **)(a1 - 8);
              v31 = &v25[v24];
            }
            else
            {
              v31 = (_QWORD *)a1;
              v25 = (_QWORD *)(a1 - v24 * 8);
            }
            if ( v31 == v25 )
            {
LABEL_31:
              v34 = v20;
              sub_AE6EC0(a4, v10);
              return v34;
            }
            v29 = v20;
            v26 = a6;
            v27 = v25;
            while ( 1 )
            {
              v33 = v26;
              v28 = sub_F947D0(*v27, a2, a3, a4, a5, v26, a7, a8, a9, a10 + 1);
              v26 = v33;
              if ( !v28 )
                break;
              v27 += 4;
              if ( v31 == v27 )
              {
                v20 = v29;
                v10 = a1;
                goto LABEL_31;
              }
            }
          }
          return 0;
        }
LABEL_33:
        if ( a7 >= v22 )
          goto LABEL_24;
        goto LABEL_21;
      }
    }
  }
  return 1;
}
