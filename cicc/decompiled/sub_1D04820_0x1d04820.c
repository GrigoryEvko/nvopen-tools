// Function: sub_1D04820
// Address: 0x1d04820
//
__int64 __fastcall sub_1D04820(__int64 a1)
{
  _QWORD *v2; // rdi
  unsigned __int8 (*v3)(void); // rax
  __int64 result; // rax
  int v5; // r15d
  unsigned int v6; // r12d
  __int64 v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // r14
  unsigned int v10; // edx
  __int64 v11; // rsi
  void (*v12)(void); // rax
  int v13; // eax
  _BYTE *v14; // r8
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD **)(a1 + 672);
  v3 = *(unsigned __int8 (**)(void))(*v2 + 64LL);
  if ( (char *)v3 == (char *)sub_1D00EA0 )
  {
    if ( v2[3] != v2[2] )
      goto LABEL_3;
  }
  else if ( !v3() )
  {
    goto LABEL_3;
  }
  *(_DWORD *)(a1 + 716) = -1;
LABEL_3:
  result = *(_QWORD *)(a1 + 680);
  v5 = (*(_QWORD *)(a1 + 688) - result) >> 3;
  if ( v5 )
  {
    v6 = 0;
    while ( 1 )
    {
      v8 = (_QWORD *)(8LL * v6 + result);
      v9 = *v8;
      if ( (*(_BYTE *)(*v8 + 236LL) & 2) == 0 )
      {
        sub_1F01F70(*v8);
        v8 = (_QWORD *)(8LL * v6 + *(_QWORD *)(a1 + 680));
      }
      v10 = *(_DWORD *)(v9 + 244);
      if ( v10 < *(_DWORD *)(a1 + 716) )
        *(_DWORD *)(a1 + 716) = v10;
      v11 = *v8;
      if ( (*(_BYTE *)(*v8 + 229LL) & 2) != 0 )
      {
        v7 = *(_QWORD *)(a1 + 672);
        if ( !byte_4FC13A0 && *(_BYTE *)(v7 + 12) )
        {
          result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 80LL))(v7);
          if ( !(_BYTE)result )
          {
            if ( v5 == ++v6 )
              return result;
            goto LABEL_9;
          }
          v7 = *(_QWORD *)(a1 + 672);
          v11 = *(_QWORD *)(*(_QWORD *)(a1 + 680) + 8LL * v6);
        }
        v12 = *(void (**)(void))(*(_QWORD *)v7 + 88LL);
        if ( (char *)v12 == (char *)sub_1D047D0 )
        {
          v13 = *(_DWORD *)(v7 + 40);
          v15[0] = v11;
          *(_DWORD *)(v7 + 40) = ++v13;
          *(_DWORD *)(v11 + 196) = v13;
          v14 = *(_BYTE **)(v7 + 24);
          if ( v14 == *(_BYTE **)(v7 + 32) )
          {
            sub_1CFD630(v7 + 16, v14, v15);
          }
          else
          {
            if ( v14 )
            {
              *(_QWORD *)v14 = v11;
              v14 = *(_BYTE **)(v7 + 24);
            }
            *(_QWORD *)(v7 + 24) = v14 + 8;
          }
        }
        else
        {
          v12();
        }
        v11 = *(_QWORD *)(*(_QWORD *)(a1 + 680) + 8LL * v6);
      }
      *(_BYTE *)(v11 + 229) &= ~1u;
      --v5;
      result = *(_QWORD *)(a1 + 680);
      *(_QWORD *)(result + 8LL * v6) = *(_QWORD *)(*(_QWORD *)(a1 + 688) - 8LL);
      *(_QWORD *)(a1 + 688) -= 8LL;
      if ( v5 == v6 )
        return result;
LABEL_9:
      result = *(_QWORD *)(a1 + 680);
    }
  }
  return result;
}
