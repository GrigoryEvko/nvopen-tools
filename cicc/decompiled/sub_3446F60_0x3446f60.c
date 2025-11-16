// Function: sub_3446F60
// Address: 0x3446f60
//
__int64 __fastcall sub_3446F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 v5; // rbx
  unsigned int v6; // eax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  int *v11; // r15
  int i; // esi
  __int64 (*v13)(); // rax
  __int64 v16; // [rsp+18h] [rbp-98h] BYREF
  __int64 v17; // [rsp+20h] [rbp-90h] BYREF
  char *v18; // [rsp+28h] [rbp-88h]
  int v19; // [rsp+30h] [rbp-80h]
  char v20; // [rsp+38h] [rbp-78h] BYREF

  v4 = 0;
  v5 = **(_QWORD **)(a2 + 40);
  v17 = sub_B2D7E0(v5, "disable-tail-calls", 0x12u);
  v6 = sub_A72A30(&v17);
  if ( !(_BYTE)v6 )
  {
    v4 = v6;
    v16 = *(_QWORD *)(v5 + 120);
    v8 = sub_A74610(&v16);
    v9 = sub_B2BE50(v5);
    v10 = v8;
    v11 = (int *)&unk_44E2160;
    sub_A74940((__int64)&v17, v9, v10);
    for ( i = 86; ; i = *v11 )
    {
      ++v11;
      sub_A77390((__int64)&v17, i);
      if ( v11 == (int *)&unk_44E2180 )
        break;
    }
    if ( !v19 && !sub_A75040((__int64)&v17, 79) && !sub_A75040((__int64)&v17, 54) )
    {
      v13 = *(__int64 (**)())(*(_QWORD *)a1 + 2336LL);
      if ( v13 != sub_302E230 )
        v4 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v13)(a1, a3, a4);
    }
    if ( v18 != &v20 )
      _libc_free((unsigned __int64)v18);
  }
  return v4;
}
