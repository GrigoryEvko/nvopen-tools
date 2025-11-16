// Function: sub_39873F0
// Address: 0x39873f0
//
void __fastcall sub_39873F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned int v5; // r12d
  unsigned __int64 *(__fastcall *v6)(__int64, __int64, unsigned __int64 *); // rax
  __int64 v7; // rsi
  __int64 (__fastcall *v8)(__int64); // rax
  void (__fastcall ***v9)(_QWORD, _QWORD, _QWORD *); // rdi
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD *); // rax
  unsigned int v11; // ebx
  int v12; // r13d
  void (__fastcall ***v13)(_QWORD, _QWORD, _QWORD *); // rdi
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD *); // rax
  __int64 v15; // rdx
  _QWORD v16[2]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v17; // [rsp+20h] [rbp-80h]
  _QWORD *v18; // [rsp+30h] [rbp-70h] BYREF
  __int64 v19; // [rsp+38h] [rbp-68h]
  _BYTE v20[16]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v21[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v22[8]; // [rsp+60h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 96);
  v18 = v20;
  v19 = 0;
  v4 = *(_QWORD *)(v3 + 504);
  v20[0] = 0;
  if ( v4 )
  {
    v5 = a2;
    v6 = *(unsigned __int64 *(__fastcall **)(__int64, __int64, unsigned __int64 *))(*(_QWORD *)v4 + 160LL);
    if ( v6 == sub_3985590 )
      sub_2241130((unsigned __int64 *)&v18, 0, 0, byte_3F871B3, 0);
    else
      v6(v4, a2, (unsigned __int64 *)&v18);
    if ( v19 )
    {
      v7 = *(_QWORD *)(a1 + 96);
      v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 392LL);
      if ( v8 == sub_215BB50 )
      {
        v21[0] = (__int64)v22;
        sub_3984920(v21, "%ERROR", (__int64)"");
      }
      else
      {
        ((void (__fastcall *)(__int64 *, __int64, _QWORD))v8)(v21, v7, v5);
      }
      v9 = *(void (__fastcall ****)(_QWORD, _QWORD, _QWORD *))(a1 + 88);
      v10 = **v9;
      v16[0] = v21;
      v11 = 1;
      v17 = 260;
      v10(v9, *(unsigned __int8 *)v18, v16);
      v12 = v19;
      if ( (_DWORD)v19 != 1 )
      {
        do
        {
          v13 = *(void (__fastcall ****)(_QWORD, _QWORD, _QWORD *))(a1 + 88);
          v14 = **v13;
          v17 = 257;
          v15 = v11++;
          v14(v13, *((unsigned __int8 *)v18 + v15), v16);
        }
        while ( v12 != v11 );
      }
      if ( (_QWORD *)v21[0] != v22 )
        j_j___libc_free_0(v21[0]);
    }
    if ( v18 != (_QWORD *)v20 )
      j_j___libc_free_0((unsigned __int64)v18);
  }
}
