// Function: sub_3028AB0
// Address: 0x3028ab0
//
__int64 __fastcall sub_3028AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // rax
  size_t v5; // rdx
  size_t v6; // rbx
  char *v7; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  char *v11; // rdx
  unsigned int *v12; // r15
  __int64 result; // rax
  char *v14; // r14
  unsigned __int64 v15; // r13
  unsigned int *v16; // rdi
  void *v17; // rdi
  void *src; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  _QWORD v20[12]; // [rsp+10h] [rbp-60h] BYREF

  v20[0] = a3;
  v20[1] = a4;
  v4 = sub_30289C0(v20, 36, (__int64)", ");
  v6 = v5;
  src = v4;
  v7 = sub_30289C0(v20, 64, (__int64)byte_3F871B3);
  v10 = v9;
  v20[4] = v7;
  v11 = v7;
  v12 = *(unsigned int **)(a1 + 8);
  result = *(_QWORD *)(a1 + 16);
  v20[5] = v10;
  v19 = result;
  if ( v12 != (unsigned int *)result )
  {
    v14 = v11;
    v15 = v10;
    while ( 1 )
    {
      v16 = v12++;
      result = sub_BC3C20(v16, a2, v11, v15);
      if ( (unsigned int *)v19 == v12 )
        break;
      v17 = *(void **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v17 >= v6 )
      {
        if ( v6 )
        {
          memcpy(v17, src, v6);
          *(_QWORD *)(a2 + 32) += v6;
        }
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)src, v6);
      }
      v11 = v14;
    }
  }
  return result;
}
