// Function: sub_1048730
// Address: 0x1048730
//
__int64 __fastcall sub_1048730(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  const void *v8; // r15
  _BYTE *v9; // r13
  _QWORD *v10; // rax
  __int64 v11; // r14
  unsigned __int8 *v12; // rax
  size_t v13; // rdx
  void *v14; // rdi
  _QWORD *v16; // rdi
  size_t v17; // [rsp+8h] [rbp-108h]
  __int64 *v18; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v19; // [rsp+20h] [rbp-F0h] BYREF
  __int64 (__fastcall **v20)(); // [rsp+28h] [rbp-E8h]
  __int64 v21; // [rsp+30h] [rbp-E0h]
  _QWORD v22[2]; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD v23[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v24; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+70h] [rbp-A0h] BYREF
  void *v26; // [rsp+80h] [rbp-90h] BYREF
  __int16 v27; // [rsp+A0h] [rbp-70h]
  void *v28[4]; // [rsp+B0h] [rbp-60h] BYREF
  char v29; // [rsp+D0h] [rbp-40h]
  char v30; // [rsp+D1h] [rbp-3Fh]

  v6 = sub_BC1CD0(a4, &unk_4F8F810, a3);
  v7 = *(_QWORD *)(v6 + 8);
  if ( *(_BYTE *)(a2 + 8) )
    sub_1047D20(*(_QWORD *)(v6 + 8));
  if ( (unsigned int)sub_2241AC0(&qword_4F8FA68, byte_3F871B3) )
  {
    v21 = v7;
    v8 = qword_4F8FA68;
    v19 = a3;
    v9 = (_BYTE *)qword_4F8FA70;
    v20 = off_49E5A18;
    v22[0] = v23;
    if ( (char *)qword_4F8FA68 + qword_4F8FA70 && !qword_4F8FA68 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v28[0] = (void *)qword_4F8FA70;
    if ( qword_4F8FA70 > 0xF )
    {
      v22[0] = sub_22409D0(v22, v28, 0);
      v16 = (_QWORD *)v22[0];
      v23[0] = v28[0];
    }
    else
    {
      if ( qword_4F8FA70 == 1 )
      {
        LOBYTE(v23[0]) = *(_BYTE *)qword_4F8FA68;
        v10 = v23;
LABEL_9:
        v22[1] = v9;
        v9[(_QWORD)v10] = 0;
        v28[0] = "MSSA";
        v27 = 257;
        v18 = &v19;
        v30 = 1;
        v29 = 3;
        sub_1048250((__int64)&v24, &v18, &v26, 0, v28, (__int64)v22);
        if ( v24 != &v25 )
          j_j___libc_free_0(v24, v25 + 1);
        if ( (_QWORD *)v22[0] != v23 )
          j_j___libc_free_0(v22[0], v23[0] + 1LL);
        v20 = off_49E5A18;
        nullsub_35();
        goto LABEL_18;
      }
      if ( !qword_4F8FA70 )
      {
        v10 = v23;
        goto LABEL_9;
      }
      v16 = v23;
    }
    memcpy(v16, v8, (size_t)v9);
    v9 = v28[0];
    v10 = (_QWORD *)v22[0];
    goto LABEL_9;
  }
  v11 = sub_904010(*(_QWORD *)a2, "MemorySSA for function: ");
  v12 = (unsigned __int8 *)sub_BD5D20(a3);
  v14 = *(void **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v14 < v13 )
  {
    v11 = sub_CB6200(v11, v12, v13);
  }
  else if ( v13 )
  {
    v17 = v13;
    memcpy(v14, v12, v13);
    *(_QWORD *)(v11 + 32) += v17;
  }
  sub_904010(v11, "\n");
  sub_103D140(v7, *(_QWORD *)a2);
LABEL_18:
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
