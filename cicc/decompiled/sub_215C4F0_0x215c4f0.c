// Function: sub_215C4F0
// Address: 0x215c4f0
//
void __fastcall sub_215C4F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // [rsp+1Ch] [rbp-54h] BYREF
  _QWORD *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v15; // [rsp+28h] [rbp-48h]
  _QWORD v16[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( !(_BYTE)a5 || (v6 = *(_DWORD *)(a3 + 48)) != 0 )
  {
    sub_215C0E0(a1, a2, a3, a4, a5);
  }
  else
  {
    v7 = HIDWORD(a5);
    if ( v7 )
    {
      LOBYTE(v16[0]) = 0;
      v14 = v16;
      v8 = *a1;
      v15 = 0;
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD **))(v8 + 160))(a1, (unsigned int)v7, &v14);
      v9 = sub_145CDC0(0x10u, a1 + 52);
      v10 = v9;
      if ( v9 )
      {
        *(_QWORD *)v9 = 0;
        *(_DWORD *)(v9 + 8) = 0;
      }
      v13 = 65547;
      sub_39A3560(a2, v9, 0, &v13, 144);
      if ( v15 )
      {
        v11 = 0;
        do
        {
          v12 = *((unsigned __int8 *)v14 + v11);
          v13 = 65547;
          sub_39A3560(a2, v10, 0, &v13, v12);
          v11 = ++v6;
        }
        while ( v6 < v15 );
      }
      sub_39A4C90(a2, a4, 2, v10);
      BYTE2(v13) = 0;
      sub_39A3560(a2, a4 + 8, 51, &v13, 2);
      if ( v14 != v16 )
        j_j___libc_free_0(v14, v16[0] + 1LL);
    }
  }
}
