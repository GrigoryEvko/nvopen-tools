// Function: sub_2354710
// Address: 0x2354710
//
void __fastcall sub_2354710(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  int v4; // r13d
  int v5; // ebx
  __int64 v6; // r15
  _QWORD *v7; // rax
  unsigned __int64 v8; // rdi
  _QWORD **v9; // rbx
  __int64 v10; // r13
  _QWORD *v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-48h]
  unsigned __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(_QWORD *)a2;
  *(_DWORD *)(a2 + 16) = 0;
  v4 = *(_DWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 12);
  *(_QWORD *)a2 = 0;
  v6 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 8) = 0;
  v12 = v2;
  v7 = (_QWORD *)sub_22077B0(0x20u);
  if ( v7 )
  {
    v7[1] = v3;
    v7[2] = v6;
    v7[3] = v12;
    v13[0] = (unsigned __int64)v7;
    *v7 = &unk_4A0F1F8;
    sub_2353900(a1, v13);
    v8 = v13[0];
    if ( !v13[0] )
    {
      _libc_free(0);
      return;
    }
    v3 = 0;
    v4 = 0;
    v5 = 0;
    goto LABEL_4;
  }
  v13[0] = 0;
  sub_2353900(a1, v13);
  v8 = v13[0];
  if ( v13[0] )
LABEL_4:
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v8 + 8LL))(v8);
  if ( v5 && v4 )
  {
    v9 = (_QWORD **)v3;
    v10 = v3 + 8LL * (unsigned int)(v4 - 1) + 8;
    do
    {
      v11 = *v9;
      if ( *v9 != (_QWORD *)-8LL )
      {
        if ( v11 )
          sub_C7D6A0((__int64)v11, *v11 + 17LL, 8);
      }
      ++v9;
    }
    while ( (_QWORD **)v10 != v9 );
  }
  _libc_free(v3);
}
