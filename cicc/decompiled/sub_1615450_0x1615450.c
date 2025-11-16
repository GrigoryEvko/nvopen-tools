// Function: sub_1615450
// Address: 0x1615450
//
void __fastcall sub_1615450(__int64 a1, __int64 a2, const char *a3, size_t a4, int a5)
{
  __int64 v6; // rdi
  __int64 v10; // rdx
  _BYTE *v11; // rdi
  _QWORD **v12; // r15
  _QWORD *v13; // rsi
  __int64 v14; // rax
  const char *v15; // rax
  size_t v16; // rdx
  const char *v17; // rsi
  void *v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-B8h]
  _QWORD **v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  size_t v25; // [rsp+18h] [rbp-A8h]
  _BYTE *v26; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-98h]
  _BYTE v28[144]; // [rsp+30h] [rbp-90h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v26 = v28;
  v27 = 0xC00000000LL;
  if ( v6 )
  {
    sub_160E770(v6, (__int64)&v26, a2);
    if ( dword_4F9EB40 > 3 )
    {
      v11 = v26;
      if ( !(_DWORD)v27 )
      {
LABEL_7:
        if ( v11 != v28 )
          _libc_free((unsigned __int64)v11);
        return;
      }
      v14 = sub_16BA580(v26, &v26, v10);
      v24 = sub_1263B40(v14, " -*- '");
      v15 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
      v17 = v15;
      v18 = *(void **)(v24 + 24);
      if ( v16 > *(_QWORD *)(v24 + 16) - (_QWORD)v18 )
      {
        v18 = (void *)v24;
        sub_16E7EE0(v24, v15);
      }
      else if ( v16 )
      {
        v22 = v24;
        v25 = v16;
        memcpy(v18, v15, v16);
        v16 = v25;
        *(_QWORD *)(v22 + 24) += v25;
      }
      v19 = sub_16BA580(v18, v17, v16);
      sub_1263B40(v19, "' is the last user of following pass instances.");
      v21 = sub_16BA580(v19, "' is the last user of following pass instances.", v20);
      sub_1263B40(v21, " Free these instances\n");
    }
    v11 = v26;
    v23 = (_QWORD **)&v26[8 * (unsigned int)v27];
    if ( v23 != (_QWORD **)v26 )
    {
      v12 = (_QWORD **)v26;
      do
      {
        v13 = *v12++;
        sub_16151B0(a1, v13, a3, a4, a5);
      }
      while ( v23 != v12 );
      v11 = v26;
    }
    goto LABEL_7;
  }
}
