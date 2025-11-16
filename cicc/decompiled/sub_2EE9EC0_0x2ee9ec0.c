// Function: sub_2EE9EC0
// Address: 0x2ee9ec0
//
__int64 __fastcall sub_2EE9EC0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  __int64 (__fastcall **v8)(); // r14
  __int64 (__fastcall ***v9)(); // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 (__fastcall ***v13)(); // rbx
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 (__fastcall **v20)(); // rax
  __int64 v21; // rdi
  __int64 (__fastcall **v22)(); // rax
  __int64 v23; // rdx
  void *v24; // rsi
  __int64 (__fastcall **v25)(); // [rsp+0h] [rbp-1F0h] BYREF
  char v26[8]; // [rsp+8h] [rbp-1E8h] BYREF
  unsigned int v27; // [rsp+10h] [rbp-1E0h]
  void *src; // [rsp+180h] [rbp-70h]
  __int64 (__fastcall **v29)(); // [rsp+188h] [rbp-68h]
  unsigned int v30; // [rsp+190h] [rbp-60h]
  char v31[8]; // [rsp+198h] [rbp-58h] BYREF
  unsigned int v32; // [rsp+1A0h] [rbp-50h]
  char v33[8]; // [rsp+1A8h] [rbp-48h] BYREF
  int v34; // [rsp+1B0h] [rbp-40h]
  __int64 (__fastcall **v35)(); // [rsp+1B8h] [rbp-38h]

  v6 = a1 + 8LL * a2 + 400;
  result = *(_QWORD *)v6;
  if ( !*(_QWORD *)v6 )
  {
    if ( a2 )
    {
      if ( a2 != 1 )
        BUG();
      v8 = off_49D4298;
      sub_2EE9920((__int64)&v25, a1, a3, a4, a5, a6);
    }
    else
    {
      v8 = off_49D42D0;
      sub_2EE9920((__int64)&v25, a1, a3, a4, a5, a6);
    }
    v25 = v8;
    v9 = (__int64 (__fastcall ***)())sub_22077B0(0x1C0u);
    v13 = v9;
    if ( v9 )
    {
      v14 = v27;
      *v9 = (__int64 (__fastcall **)())&unk_4A2A380;
      v9[1] = (__int64 (__fastcall **)())(v9 + 3);
      v9[2] = (__int64 (__fastcall **)())0x400000000LL;
      if ( (_DWORD)v14 )
        sub_2EE9BA0((__int64)(v9 + 1), (__int64)v26, v10, v14, v11, v12);
      v13[47] = 0;
      v13[48] = 0;
      v13[49] = 0;
      *((_DWORD *)v13 + 100) = 0;
      sub_C7D6A0(0, 0, 8);
      v18 = v30;
      *((_DWORD *)v13 + 100) = v30;
      if ( (_DWORD)v18 )
      {
        v22 = (__int64 (__fastcall **)())sub_C7D670(16 * v18, 8);
        v23 = *((unsigned int *)v13 + 100);
        v24 = src;
        v13[48] = v22;
        v13[49] = v29;
        memcpy(v22, v24, 16 * v23);
      }
      else
      {
        v13[48] = 0;
        v13[49] = 0;
      }
      v19 = v32;
      v13[52] = 0;
      v13[51] = (__int64 (__fastcall **)())(v13 + 53);
      if ( (_DWORD)v19 )
        sub_2EE7570((__int64)(v13 + 51), (__int64)v31, v19, v15, v16, v17);
      v13[54] = 0;
      v13[53] = (__int64 (__fastcall **)())(v13 + 55);
      if ( v34 )
        sub_2EE7570((__int64)(v13 + 53), (__int64)v33, v19, v15, v16, v17);
      v20 = v35;
      *v13 = v8;
      v13[55] = v20;
    }
    v21 = *(_QWORD *)v6;
    *(_QWORD *)v6 = v13;
    if ( v21 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 24LL))(v21);
    v25 = v8;
    sub_2EE8570((__int64)&v25);
    return *(_QWORD *)v6;
  }
  return result;
}
