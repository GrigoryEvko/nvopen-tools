// Function: sub_1EE96B0
// Address: 0x1ee96b0
//
__int64 __fastcall sub_1EE96B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        char a8)
{
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  bool v13; // zf
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 result; // rax
  unsigned int v17; // ebx
  __int64 v18; // rax
  int v21[13]; // [rsp+2Ch] [rbp-34h] BYREF

  sub_1EE6140(a1);
  v11 = 0;
  *(_QWORD *)a1 = a2;
  v12 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v12 != sub_1D00B10 )
  {
    v18 = ((__int64 (__fastcall *)(_QWORD))v12)(*(_QWORD *)(a2 + 16));
    a2 = *(_QWORD *)a1;
    v11 = v18;
  }
  v13 = *(_BYTE *)(a1 + 56) == 0;
  *(_QWORD *)(a1 + 8) = v11;
  *(_QWORD *)(a1 + 16) = a3;
  v14 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 24) = v14;
  *(_BYTE *)(a1 + 57) = a8;
  *(_BYTE *)(a1 + 58) = a7;
  if ( !v13 )
    *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 64) = a6;
  v21[0] = 0;
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 200LL))(v11);
  sub_1D05C60(a1 + 72, v15, v21);
  sub_1EE5340(*(_QWORD *)(a1 + 48), (char **)(a1 + 72));
  result = sub_1EE6070(a1 + 96, *(_DWORD **)(a1 + 24));
  if ( a8 )
  {
    v17 = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
    result = *(unsigned int *)(a1 + 256);
    if ( v17 < *(_DWORD *)(a1 + 256) >> 2 || v17 > (unsigned int)result )
    {
      _libc_free(*(_QWORD *)(a1 + 248));
      result = (__int64)_libc_calloc(v17, 1u);
      if ( !result )
      {
        if ( v17 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          result = 0;
        }
        else
        {
          result = sub_13A3880(1u);
        }
      }
      *(_DWORD *)(a1 + 256) = v17;
      *(_QWORD *)(a1 + 248) = result;
    }
  }
  return result;
}
