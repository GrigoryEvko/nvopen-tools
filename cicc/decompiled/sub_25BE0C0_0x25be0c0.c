// Function: sub_25BE0C0
// Address: 0x25be0c0
//
__int64 __fastcall sub_25BE0C0(__int64 a1, unsigned __int64 a2)
{
  __int8 *v2; // rbx
  __int64 v3; // rdx
  __m128i *v4; // r13
  __int64 v5; // r9
  int v6; // eax
  __int8 *v7; // r13
  void (__fastcall *v8)(__int8 *, __int8 *, __int64); // rax
  void (__fastcall *v9)(__int8 *, __int8 *, __int64); // rax
  void (__fastcall *v10)(__int8 *, __int8 *, __int64); // rax
  __int64 result; // rax
  __int64 v12; // r14
  unsigned __int64 v13; // rbx
  int v14; // r15d
  int v15; // r15d
  unsigned __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (__int8 *)a2;
  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(__m128i **)a1;
  v5 = v3 + 1;
  v6 = *(_DWORD *)(a1 + 8);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v12 = a1 + 16;
    if ( (unsigned __int64)v4 > a2 || a2 >= (unsigned __int64)v4 + 104 * v3 )
    {
      v4 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v3 + 1, 0x68u, v16, v5);
      sub_25BD6E0(a1, v4);
      v15 = v16[0];
      if ( v12 != *(_QWORD *)a1 )
        _libc_free(*(_QWORD *)a1);
      v3 = *(unsigned int *)(a1 + 8);
      *(_QWORD *)a1 = v4;
      *(_DWORD *)(a1 + 12) = v15;
      v6 = v3;
    }
    else
    {
      v13 = a2 - (_QWORD)v4;
      v4 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v3 + 1, 0x68u, v16, v5);
      sub_25BD6E0(a1, v4);
      v14 = v16[0];
      if ( v12 != *(_QWORD *)a1 )
        _libc_free(*(_QWORD *)a1);
      v3 = *(unsigned int *)(a1 + 8);
      *(_QWORD *)a1 = v4;
      v2 = &v4->m128i_i8[v13];
      *(_DWORD *)(a1 + 12) = v14;
      v6 = v3;
    }
  }
  v7 = &v4->m128i_i8[104 * v3];
  if ( v7 )
  {
    *((_QWORD *)v7 + 2) = 0;
    v8 = (void (__fastcall *)(__int8 *, __int8 *, __int64))*((_QWORD *)v2 + 2);
    if ( v8 )
    {
      v8(v7, v2, 2);
      *((_QWORD *)v7 + 3) = *((_QWORD *)v2 + 3);
      *((_QWORD *)v7 + 2) = *((_QWORD *)v2 + 2);
    }
    *((_QWORD *)v7 + 6) = 0;
    v9 = (void (__fastcall *)(__int8 *, __int8 *, __int64))*((_QWORD *)v2 + 6);
    if ( v9 )
    {
      v9(v7 + 32, v2 + 32, 2);
      *((_QWORD *)v7 + 7) = *((_QWORD *)v2 + 7);
      *((_QWORD *)v7 + 6) = *((_QWORD *)v2 + 6);
    }
    *((_QWORD *)v7 + 10) = 0;
    v10 = (void (__fastcall *)(__int8 *, __int8 *, __int64))*((_QWORD *)v2 + 10);
    if ( v10 )
    {
      v10(v7 + 64, v2 + 64, 2);
      *((_QWORD *)v7 + 11) = *((_QWORD *)v2 + 11);
      *((_QWORD *)v7 + 10) = *((_QWORD *)v2 + 10);
    }
    *((_DWORD *)v7 + 24) = *((_DWORD *)v2 + 24);
    v7[100] = v2[100];
    v6 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v6 + 1);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
