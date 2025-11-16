// Function: sub_29D66F0
// Address: 0x29d66f0
//
bool __fastcall sub_29D66F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  int v5; // edi
  unsigned __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // rsi
  unsigned __int64 v13; // rax
  int v14; // ecx
  unsigned __int64 v15; // rax
  _QWORD *v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // edx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int8 *v22; // r12
  __int64 v23; // rsi
  __int64 v25; // r15
  unsigned __int64 v26; // r14
  __int64 v27; // r13
  unsigned __int8 *v28; // r15
  __int64 v29; // rdx
  __int64 v30; // r12
  __int64 v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v33; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+28h] [rbp-B8h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v37; // [rsp+38h] [rbp-A8h]
  __int64 v39; // [rsp+48h] [rbp-98h]
  _QWORD v40[6]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v41[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a4 + 48 )
  {
    v34 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = *(unsigned __int8 *)(v4 - 24);
    v6 = v4 - 24;
    v7 = (unsigned int)(v5 - 30) < 0xB;
    v8 = 0;
    if ( v7 )
      v8 = v6;
    v34 = v8;
  }
  v9 = *(_QWORD *)(a4 + 56);
  v10 = *(_QWORD *)(a2 + 56);
  v11 = v9 - 24;
  if ( !v9 )
    v11 = 0;
  v12 = (_QWORD *)(a2 + 48);
  v36 = v11;
  v13 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v13 == v12 )
  {
    v15 = 0;
  }
  else
  {
    if ( !v13 )
      BUG();
    v14 = *(unsigned __int8 *)(v13 - 24);
    v15 = v13 - 24;
    if ( (unsigned int)(v14 - 30) >= 0xB )
      v15 = 0;
  }
  v16 = (_QWORD *)(a3 + 48);
  v37 = v15 + 24;
  v39 = v16[1];
  v17 = *v16 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v17 == v16 )
  {
    v33 = 0;
  }
  else
  {
    if ( !v17 )
      BUG();
    v18 = *(unsigned __int8 *)(v17 - 24);
    v19 = v17 - 24;
    v7 = (unsigned int)(v18 - 30) < 0xB;
    v20 = 0;
    if ( v7 )
      v20 = v19;
    v33 = v20;
  }
  if ( v37 == v10 )
    return v39 == v33 + 24;
  v21 = 0;
  if ( v36 )
    v21 = v36 + 24;
  v35 = v21;
  while ( 1 )
  {
    v22 = (unsigned __int8 *)(v10 - 24);
    if ( !v10 )
      v22 = 0;
    v23 = v39 - 24;
    if ( !v39 )
      v23 = 0;
    if ( !sub_B46220((__int64)v22, v23)
      || (unsigned __int8)sub_B46970(v22) && (*v22 != 62 || (v22[2] & 1) != 0)
      || (unsigned __int8)sub_B46420((__int64)v22) )
    {
      return 0;
    }
    if ( (unsigned __int8)sub_B46490((__int64)v22) )
    {
      if ( v36 )
      {
        if ( v34 )
        {
          v25 = v35;
          v26 = v34 + 24;
          if ( v34 + 24 != v35 )
            goto LABEL_37;
        }
        else
        {
          v32 = v10;
          v26 = 0;
          v27 = v36 + 24;
          v28 = v22;
          do
          {
LABEL_38:
            v29 = v27 - 24;
            if ( !v27 )
              v29 = 0;
            v30 = v29;
            if ( (unsigned __int8)sub_B46420(v29) || (unsigned __int8)sub_B46490(v30) )
            {
              v31 = *a1;
              if ( !*a1 )
                return 0;
              v41[0] = v30;
              v41[1] = -1;
              memset(&v41[2], 0, 32);
              v40[0] = v28;
              v40[1] = -1;
              memset(&v40[2], 0, 32);
              if ( (unsigned __int8)sub_CF4E00(v31, (__int64)v40, (__int64)v41) )
                return 0;
            }
            v27 = *(_QWORD *)(v27 + 8);
          }
          while ( v27 != v26 );
          v10 = v32;
        }
      }
      else
      {
        v25 = v35;
        v26 = v34 + 24;
        if ( v34 )
        {
LABEL_37:
          v32 = v10;
          v27 = v25;
          v28 = v22;
          goto LABEL_38;
        }
      }
    }
    v10 = *(_QWORD *)(v10 + 8);
    v39 = *(_QWORD *)(v39 + 8);
    if ( v37 == v10 )
      return v39 == v33 + 24;
  }
}
