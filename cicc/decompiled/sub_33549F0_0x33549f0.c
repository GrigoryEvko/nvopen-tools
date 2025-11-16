// Function: sub_33549F0
// Address: 0x33549f0
//
__int64 __fastcall sub_33549F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(__int64 a1, __int64 a2),
        __int64 a5,
        __int64 a6)
{
  _QWORD *v7; // rdi
  unsigned __int8 (*v8)(void); // rax
  __int64 result; // rax
  int v10; // r15d
  unsigned int v11; // r12d
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // r14
  unsigned int v16; // edx
  void (*v17)(void); // rax
  int v18; // eax
  _BYTE *v19; // r8
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_QWORD **)(a1 + 640);
  v8 = *(unsigned __int8 (**)(void))(*v7 + 64LL);
  if ( (char *)v8 == (char *)sub_3351550 )
  {
    if ( v7[3] != v7[2] )
      goto LABEL_3;
  }
  else if ( !v8() )
  {
    goto LABEL_3;
  }
  *(_DWORD *)(a1 + 684) = -1;
LABEL_3:
  result = *(_QWORD *)(a1 + 648);
  v10 = (*(_QWORD *)(a1 + 656) - result) >> 3;
  if ( v10 )
  {
    v11 = 0;
    while ( 1 )
    {
      v13 = 8LL * v11;
      v14 = (__int64 *)(v13 + result);
      v15 = *v14;
      if ( (*(_BYTE *)(*v14 + 254) & 2) == 0 )
      {
        sub_2F8F770(*v14, (_QWORD *)a2, v13, (__int64)a4, a5, a6);
        v14 = (__int64 *)(8LL * v11 + *(_QWORD *)(a1 + 648));
      }
      v16 = *(_DWORD *)(v15 + 244);
      if ( v16 < *(_DWORD *)(a1 + 684) )
        *(_DWORD *)(a1 + 684) = v16;
      a2 = *v14;
      if ( (*(_BYTE *)(*v14 + 249) & 2) != 0 )
      {
        v12 = *(_QWORD *)(a1 + 640);
        if ( !byte_5038F08 && *(_BYTE *)(v12 + 12) )
        {
          result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 80LL))(v12);
          if ( !(_BYTE)result )
          {
            if ( v10 == ++v11 )
              return result;
            goto LABEL_9;
          }
          v12 = *(_QWORD *)(a1 + 640);
          a2 = *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8LL * v11);
        }
        a4 = sub_33549A0;
        v17 = *(void (**)(void))(*(_QWORD *)v12 + 88LL);
        if ( (char *)v17 == (char *)sub_33549A0 )
        {
          v18 = *(_DWORD *)(v12 + 40);
          v20[0] = a2;
          *(_DWORD *)(v12 + 40) = ++v18;
          *(_DWORD *)(a2 + 204) = v18;
          v19 = *(_BYTE **)(v12 + 24);
          if ( v19 == *(_BYTE **)(v12 + 32) )
          {
            sub_2ECAD30(v12 + 16, v19, v20);
          }
          else
          {
            if ( v19 )
            {
              *(_QWORD *)v19 = a2;
              v19 = *(_BYTE **)(v12 + 24);
            }
            a5 = (__int64)(v19 + 8);
            *(_QWORD *)(v12 + 24) = a5;
          }
        }
        else
        {
          v17();
        }
        a2 = *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8LL * v11);
      }
      *(_BYTE *)(a2 + 249) &= ~1u;
      --v10;
      result = *(_QWORD *)(a1 + 648);
      *(_QWORD *)(result + 8LL * v11) = *(_QWORD *)(*(_QWORD *)(a1 + 656) - 8LL);
      *(_QWORD *)(a1 + 656) -= 8LL;
      if ( v10 == v11 )
        return result;
LABEL_9:
      result = *(_QWORD *)(a1 + 648);
    }
  }
  return result;
}
