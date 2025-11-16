// Function: sub_2310860
// Address: 0x2310860
//
_QWORD *__fastcall sub_2310860(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  int v8; // eax
  unsigned __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rbx
  _QWORD *v12; // r13
  _QWORD *v13; // r14
  __int64 v14; // r8
  __int64 (__fastcall *v15)(_QWORD *); // rax
  unsigned __int64 v16; // rdi
  _QWORD **v18; // r14
  __int64 v19; // r15
  unsigned __int64 v20; // rdi
  __int64 v21; // r10
  _QWORD *v22; // r8
  _QWORD *v23; // r9
  __int64 (__fastcall *v24)(_QWORD *); // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  int v26; // [rsp+10h] [rbp-60h]
  _QWORD *v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  int v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  _QWORD *v34; // [rsp+18h] [rbp-58h]
  _QWORD *v35; // [rsp+18h] [rbp-58h]
  _QWORD *v36; // [rsp+18h] [rbp-58h]
  unsigned __int64 v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39; // [rsp+30h] [rbp-40h]

  sub_2DCFEE0(&v37, a2 + 8);
  v3 = v37;
  v4 = v38;
  v5 = v39;
  v37 = 0;
  v30 = v38;
  LODWORD(v39) = 0;
  v26 = HIDWORD(v38);
  v38 = 0;
  v6 = (_QWORD *)sub_22077B0(0x20u);
  v7 = v6;
  if ( v6 )
  {
    v6[2] = v4;
    v6[3] = v5;
    v6[1] = v3;
    v3 = 0;
    *v6 = &unk_4A0B498;
  }
  else if ( v26 && v30 )
  {
    v18 = (_QWORD **)v3;
    v19 = v3 + 8LL * (unsigned int)(v30 - 1) + 8;
    do
    {
      v22 = *v18;
      if ( *v18 != (_QWORD *)-8LL && v22 )
      {
        v23 = (_QWORD *)v22[1];
        v21 = *v22 + 17LL;
        if ( v23 )
        {
          v24 = *(__int64 (__fastcall **)(_QWORD *))(*v23 + 8LL);
          if ( v24 == sub_BD9990 )
          {
            v20 = v23[1];
            *v23 = &unk_49DB390;
            if ( (_QWORD *)v20 != v23 + 3 )
            {
              v25 = v21;
              v27 = v22;
              v34 = v23;
              j_j___libc_free_0(v20);
              v21 = v25;
              v22 = v27;
              v23 = v34;
            }
            v28 = v21;
            v35 = v22;
            j_j___libc_free_0((unsigned __int64)v23);
            v22 = v35;
            v21 = v28;
          }
          else
          {
            v29 = *v22 + 17LL;
            v36 = *v18;
            v24(v23);
            v21 = v29;
            v22 = v36;
          }
        }
        sub_C7D6A0((__int64)v22, v21, 8);
      }
      ++v18;
    }
    while ( (_QWORD **)v19 != v18 );
  }
  _libc_free(v3);
  v8 = HIDWORD(v38);
  *a1 = v7;
  if ( v8 )
  {
    v9 = v37;
    if ( (_DWORD)v38 )
    {
      v10 = 8LL * (unsigned int)v38;
      v11 = 0;
      do
      {
        v12 = *(_QWORD **)(v9 + v11);
        if ( v12 && v12 != (_QWORD *)-8LL )
        {
          v13 = (_QWORD *)v12[1];
          v14 = *v12 + 17LL;
          if ( v13 )
          {
            v15 = *(__int64 (__fastcall **)(_QWORD *))(*v13 + 8LL);
            if ( v15 == sub_BD9990 )
            {
              v16 = v13[1];
              *v13 = &unk_49DB390;
              if ( (_QWORD *)v16 != v13 + 3 )
              {
                v31 = v14;
                j_j___libc_free_0(v16);
                v14 = v31;
              }
              v32 = v14;
              j_j___libc_free_0((unsigned __int64)v13);
              v14 = v32;
            }
            else
            {
              v33 = *v12 + 17LL;
              v15(v13);
              v14 = v33;
            }
          }
          sub_C7D6A0((__int64)v12, v14, 8);
          v9 = v37;
        }
        v11 += 8;
      }
      while ( v10 != v11 );
    }
  }
  else
  {
    v9 = v37;
  }
  _libc_free(v9);
  return a1;
}
