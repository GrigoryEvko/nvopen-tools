// Function: sub_222BCC0
// Address: 0x222bcc0
//
bool __fastcall sub_222BCC0(__int64 a1, char *a2, size_t a3, __int64 a4)
{
  __int64 v5; // rdi
  size_t v6; // r12
  int v7; // eax
  __int64 v8; // rdi
  size_t v9; // rdx
  void *v10; // rsp
  unsigned int v11; // eax
  size_t v12; // rax
  char v14[4]; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+4h] [rbp-5Ch]
  __int64 v16; // [rsp+8h] [rbp-58h]
  _QWORD *v17; // [rsp+10h] [rbp-50h]
  __int64 *v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(_QWORD *)(a1 + 200);
  if ( !v5 )
    sub_426219(0, a2, a3, a4);
  v6 = a3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL))(v5) )
    goto LABEL_5;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 64LL))(*(_QWORD *)(a1 + 200));
  v8 = *(_QWORD *)(a1 + 200);
  v18 = &v19;
  v9 = v6 * v7;
  v17 = v20;
  v10 = alloca(v9 + 8);
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, char *, char *, _QWORD *, char *, char *, __int64 *))(*(_QWORD *)v8 + 16LL))(
          v8,
          a1 + 132,
          a2,
          &a2[v6],
          v20,
          v14,
          &v14[v9],
          &v19);
  if ( v11 > 1 )
  {
    if ( v11 == 3 )
    {
LABEL_5:
      v12 = sub_2207DF0((FILE **)(a1 + 104), a2, v6);
      return v12 == v6;
    }
LABEL_11:
    sub_426A1E((__int64)"basic_filebuf::_M_convert_to_external conversion error");
  }
  v15 = v11;
  v16 = v19;
  v6 = v19 - (_QWORD)v14;
  v12 = sub_2207DF0((FILE **)(a1 + 104), v14, v19 - (_QWORD)v14);
  if ( v6 == v12 && (v15 & 1) != 0 )
  {
    if ( (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD *, char *, __int64, __int64 *))(**(_QWORD **)(a1 + 200) + 16LL))(
           *(_QWORD *)(a1 + 200),
           a1 + 132,
           v20[0],
           *(_QWORD *)(a1 + 40),
           v17,
           v14,
           v16,
           v18) != 2 )
    {
      v6 = v19 - (_QWORD)v14;
      v12 = sub_2207DF0((FILE **)(a1 + 104), v14, v19 - (_QWORD)v14);
      return v12 == v6;
    }
    goto LABEL_11;
  }
  return v12 == v6;
}
