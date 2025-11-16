// Function: sub_2D5AAA0
// Address: 0x2d5aaa0
//
__int64 __fastcall sub_2D5AAA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rbx
  _QWORD *v5; // rdx
  _BYTE *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // r15d
  __int64 (*v13)(); // rax
  unsigned int v14; // r15d
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v32[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 || *(_QWORD *)(v4 + 8) || *(_QWORD *)(a1 + 40) != *(_QWORD *)(*(_QWORD *)(v4 + 24) + 40LL) )
  {
    v5 = (*(_BYTE *)(a1 + 7) & 0x40) != 0
       ? *(_QWORD **)(a1 - 8)
       : (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v6 = (_BYTE *)*v5;
    if ( *(_BYTE *)*v5 == 17
      || (v7 = v5[4], *(_BYTE *)v7 == 17)
      || (v8 = *((_QWORD *)v6 + 2)) == 0
      || *(_QWORD *)(v8 + 8)
      || (v16 = *(_QWORD *)(v7 + 16)) == 0
      || *(_QWORD *)(v16 + 8) )
    {
      for ( ; v4; v4 = *(_QWORD *)(v4 + 8) )
      {
        v9 = *(_QWORD *)(v4 + 24);
        if ( *(_BYTE *)v9 != 82 )
          return 0;
        v10 = (*(_BYTE *)(v9 + 7) & 0x40) != 0 ? *(_QWORD *)(v9 - 8) : v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
        v11 = *(_QWORD *)(v10 + 32);
        if ( *(_BYTE *)v11 != 17 )
          return 0;
        v12 = *(_DWORD *)(v11 + 32);
        if ( v12 <= 0x40 )
        {
          if ( *(_QWORD *)(v11 + 24) )
            return 0;
        }
        else if ( v12 != (unsigned int)sub_C444A0(v11 + 24) )
        {
          return 0;
        }
      }
      v13 = *(__int64 (**)())(*(_QWORD *)a2 + 352LL);
      if ( v13 != sub_2D565E0 )
      {
        v14 = ((__int64 (__fastcall *)(__int64, __int64))v13)(a2, a1);
        if ( (_BYTE)v14 )
        {
          v17 = *(_QWORD *)(a1 + 16);
          if ( !v17 )
          {
LABEL_46:
            sub_B43D60((_QWORD *)a1);
            return v14;
          }
          while ( 1 )
          {
            v20 = *(_QWORD *)(v17 + 24);
            v31 = v17;
            v17 = *(_QWORD *)(v17 + 8);
            if ( *(_QWORD *)(v20 + 40) == *(_QWORD *)(a1 + 40) )
              v20 = a1;
            v33 = 257;
            LOWORD(v2) = 0;
            v28 = v20 + 24;
            v21 = (__int64 *)sub_986520(a1);
            v22 = sub_B504D0(28, *v21, v21[4], (__int64)v32, v28, v2);
            v23 = *(_QWORD *)(a1 + 48);
            v18 = v22;
            v32[0] = v23;
            if ( v23 )
              break;
            v19 = v22 + 48;
            if ( (__int64 *)(v22 + 48) != v32 )
            {
              v24 = *(_QWORD *)(v22 + 48);
              if ( v24 )
                goto LABEL_41;
            }
LABEL_35:
            sub_AC2B30(v31, v18);
            if ( !v17 )
              goto LABEL_46;
          }
          v27 = v22;
          sub_B96E90((__int64)v32, v23, 1);
          v18 = v27;
          v19 = v27 + 48;
          if ( (__int64 *)(v27 + 48) == v32 )
          {
            if ( v32[0] )
            {
              sub_B91220((__int64)v32, v32[0]);
              v18 = v27;
            }
            goto LABEL_35;
          }
          v24 = *(_QWORD *)(v27 + 48);
          if ( v24 )
          {
LABEL_41:
            v26 = v18;
            v29 = v19;
            sub_B91220(v19, v24);
            v18 = v26;
            v19 = v29;
          }
          v25 = (unsigned __int8 *)v32[0];
          *(_QWORD *)(v18 + 48) = v32[0];
          if ( v25 )
          {
            v30 = v18;
            sub_B976B0((__int64)v32, v25, v19);
            v18 = v30;
          }
          goto LABEL_35;
        }
      }
    }
  }
  return 0;
}
