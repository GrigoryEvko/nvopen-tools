// Function: sub_25284B0
// Address: 0x25284b0
//
__int64 __fastcall sub_25284B0(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 result; // rax
  int v9; // edx
  int v10; // r8d
  void *v11; // rdi
  __int64 v12; // [rsp-10h] [rbp-30h]
  _QWORD v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v13[0] = a3;
  if ( a4 || (result = sub_A73170(v13, 89), !(_BYTE)result) )
  {
    v6 = *(_QWORD *)(a1 + 4376);
    if ( !v6 )
      goto LABEL_5;
    v7 = *(_QWORD *)(v6 + 8);
    result = *(unsigned int *)(v6 + 24);
    if ( !(_DWORD)result )
      return result;
    v9 = result - 1;
    v10 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)&unk_438A66B >> 9) ^ ((unsigned int)&unk_438A66B >> 4));
    v11 = *(void **)(v7 + 8 * result);
    if ( v11 != &unk_438A66B )
    {
      while ( v11 != (void *)-4096LL )
      {
        result = v9 & (unsigned int)(v10 + result);
        v11 = *(void **)(v7 + 8LL * (unsigned int)result);
        if ( v11 == &unk_438A66B )
          goto LABEL_5;
        ++v10;
      }
    }
    else
    {
LABEL_5:
      result = sub_2553E90(a1, a2, 89, 0);
      if ( !(_BYTE)result )
      {
        sub_2527F10(a1, *a2, a2[1], 0, 2, 0, 1);
        return v12;
      }
    }
  }
  return result;
}
