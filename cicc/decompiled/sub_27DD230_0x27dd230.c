// Function: sub_27DD230
// Address: 0x27dd230
//
__int64 __fastcall sub_27DD230(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v5; // r15
  __int64 v6; // rdx
  unsigned int v7; // r13d
  __int64 i; // r14
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rcx
  __int16 v12; // ax
  bool v13; // cl
  __int64 v14; // rax
  __int64 result; // rax
  _BYTE *v16; // rax
  bool v17; // r11
  __int64 v18; // r13
  __int64 v19; // r14
  _QWORD *v20; // rax
  bool v21; // r11
  _QWORD *v22; // r15
  __int64 v23; // rsi
  unsigned __int64 *v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdi
  __int64 v29; // rdi
  _QWORD *v30; // rax
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  bool v33; // [rsp+Fh] [rbp-61h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  bool v35; // [rsp+10h] [rbp-60h]
  bool v36; // [rsp+10h] [rbp-60h]
  _BYTE *v37; // [rsp+18h] [rbp-58h]
  bool v38; // [rsp+18h] [rbp-58h]
  bool v39; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v40; // [rsp+18h] [rbp-58h]
  bool v41; // [rsp+18h] [rbp-58h]
  _QWORD *v42; // [rsp+20h] [rbp-50h]
  unsigned __int64 v44[8]; // [rsp+30h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_52;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_52:
    BUG();
  if ( *(_BYTE *)(v2 - 24) == 31 && (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) == 3 )
  {
    v3 = *(_QWORD *)(v2 - 120);
    v42 = 0;
    v37 = (_BYTE *)v3;
    if ( *(_BYTE *)v3 == 96 )
    {
      v30 = *(_QWORD **)(v3 + 16);
      v42 = v30;
      if ( v30 )
      {
        if ( v30[1] )
        {
          v42 = 0;
        }
        else
        {
          v42 = *(_QWORD **)(v2 - 120);
          v37 = (_BYTE *)*((_QWORD *)v37 - 4);
        }
      }
    }
    v5 = sub_AA54C0(a2);
    v34 = sub_AA4E30(a2);
    if ( v5 )
    {
      v6 = a2;
      v7 = 0;
      for ( i = v5; ; i = v14 )
      {
        if ( v7 >= (unsigned int)qword_4FFDB48 )
          return 0;
        v9 = *(_QWORD *)(i + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v9 == i + 48 )
          goto LABEL_53;
        if ( !v9 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_53:
          BUG();
        if ( *(_BYTE *)(v9 - 24) != 31 || (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 3 )
          return 0;
        v10 = *(_QWORD *)(v9 - 56);
        if ( !v10 || v6 != v10 )
        {
          v11 = *(_QWORD *)(v9 - 88);
          if ( v6 != v11 || !v11 )
            return 0;
        }
        v33 = v6 == v10;
        v12 = sub_9A18B0(*(_QWORD *)(v9 - 120), v37, v34, v6 == v10, 0);
        v13 = (v42 != 0) & (HIBYTE(v12) ^ 1);
        if ( v13 )
        {
          v16 = *(_BYTE **)(v9 - 120);
          if ( *v16 == 96 && *(v42 - 4) == *((_QWORD *)v16 - 4) )
          {
            v17 = v13;
            goto LABEL_28;
          }
        }
        else if ( HIBYTE(v12) )
        {
          v33 = v12;
          v17 = v42 != 0;
LABEL_28:
          v38 = v17;
          v18 = *(_QWORD *)(v2 + -32 - 32LL * !v33 - 24);
          v19 = *(_QWORD *)(v2 + -32 - 32LL * v33 - 24);
          sub_AA5980(v19, a2, 0);
          v20 = sub_BD2C40(72, 1u);
          v21 = v38;
          v22 = v20;
          if ( v20 )
          {
            sub_B4C8F0((__int64)v20, v18, 1u, v2, 0);
            v21 = v38;
          }
          v23 = *(_QWORD *)(v2 + 24);
          v24 = v22 + 6;
          v44[0] = v23;
          if ( v23 )
          {
            v35 = v21;
            sub_B96E90((__int64)v44, v23, 1);
            v24 = v22 + 6;
            v21 = v35;
            if ( v22 + 6 == v44 )
            {
              if ( v44[0] )
              {
                sub_B91220((__int64)v44, v44[0]);
                v21 = v35;
              }
              goto LABEL_34;
            }
            v31 = v22[6];
            if ( !v31 )
            {
LABEL_45:
              v32 = (unsigned __int8 *)v44[0];
              v22[6] = v44[0];
              if ( v32 )
              {
                v41 = v21;
                sub_B976B0((__int64)v44, v32, (__int64)v24);
                v21 = v41;
              }
              goto LABEL_34;
            }
          }
          else if ( v24 == v44 || (v31 = v22[6]) == 0 )
          {
LABEL_34:
            v39 = v21;
            sub_B43D60((_QWORD *)(v2 - 24));
            if ( v39 )
              sub_B43D60(v42);
            v44[0] = a2;
            v28 = a1[6];
            v44[1] = v19 | 4;
            sub_FFDB80(v28, v44, 1, v25, v26, v27);
            v29 = sub_27DD130(a1);
            result = 1;
            if ( v29 )
            {
              sub_FF0C10(v29, a2);
              return 1;
            }
            return result;
          }
          v36 = v21;
          v40 = v24;
          sub_B91220((__int64)v24, v31);
          v21 = v36;
          v24 = v40;
          goto LABEL_45;
        }
        ++v7;
        v14 = sub_AA54C0(i);
        v6 = i;
        if ( !v14 )
          return 0;
      }
    }
  }
  return 0;
}
