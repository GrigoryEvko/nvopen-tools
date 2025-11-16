// Function: sub_B81AB0
// Address: 0xb81ab0
//
__int64 __fastcall sub_B81AB0(__int64 a1, _QWORD *a2, const void *a3, size_t a4, int a5)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 result; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // r8
  unsigned int v14; // r9d
  _QWORD v15[12]; // [rsp+0h] [rbp-60h] BYREF

  sub_B817B0(a1, (__int64)a2, 2, a5, a3, a4);
  sub_C85EE0(v15);
  v15[2] = a2;
  v15[3] = 0;
  v15[4] = 0;
  v15[0] = &unk_49DA748;
  v6 = sub_BC4450(a2);
  if ( v6 )
  {
    v7 = v6;
    sub_C9E250(v6);
    (*(void (__fastcall **)(_QWORD *))(*a2 + 96LL))(a2);
    sub_C9E2A0(v7);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD *))(*a2 + 96LL))(a2);
  }
  v15[0] = &unk_49DA748;
  nullsub_162(v15);
  result = *(unsigned int *)(a1 + 232);
  v9 = a2[2];
  v10 = *(_QWORD *)(a1 + 216);
  if ( (_DWORD)result )
  {
    v11 = result - 1;
    v12 = (result - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    result = v10 + 16LL * v12;
    v13 = *(_QWORD *)result;
    if ( v9 == *(_QWORD *)result )
    {
LABEL_5:
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(a1 + 224);
      ++*(_DWORD *)(a1 + 228);
    }
    else
    {
      result = 1;
      while ( v13 != -4096 )
      {
        v14 = result + 1;
        v12 = v11 & (result + v12);
        result = v10 + 16LL * v12;
        v13 = *(_QWORD *)result;
        if ( v9 == *(_QWORD *)result )
          goto LABEL_5;
        result = v14;
      }
    }
  }
  return result;
}
