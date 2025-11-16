// Function: sub_1436EA0
// Address: 0x1436ea0
//
__int64 __fastcall sub_1436EA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // eax
  __int64 *v5; // rdx
  __int64 *v6; // r12
  __int64 *v7; // r13
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+10h] [rbp-30h]
  int v21; // [rsp+18h] [rbp-28h]

  v3 = **(_QWORD **)(a2 + 32);
  *(_WORD *)a1 = 0;
  v4 = sub_14AE980(v3) ^ 1;
  *(_BYTE *)(a1 + 1) = v4;
  *(_BYTE *)a1 = v4;
  v5 = *(__int64 **)(a2 + 32);
  v6 = *(__int64 **)(a2 + 40);
  v7 = v5 + 1;
  if ( v6 != v5 + 1 )
  {
    do
    {
      if ( (_BYTE)v4 )
        break;
      v8 = *v7++;
      LOBYTE(v4) = *(_BYTE *)a1 | sub_14AE980(v8) ^ 1;
      *(_BYTE *)a1 = v4;
    }
    while ( v7 != v6 );
    v5 = *(__int64 **)(a2 + 32);
  }
  result = *v5;
  v10 = *(_QWORD *)(*v5 + 56);
  if ( (*(_BYTE *)(v10 + 18) & 8) != 0 )
  {
    result = sub_15E38F0(*(_QWORD *)(*v5 + 56));
    if ( result )
    {
      result = sub_14DD7D0(result);
      if ( (int)result > 10 )
      {
        if ( (_DWORD)result != 12 )
          return result;
      }
      else if ( (int)result <= 6 )
      {
        return result;
      }
      sub_14DDFC0(&v18, v10);
      v11 = *(unsigned int *)(a1 + 32);
      if ( (_DWORD)v11 )
      {
        v12 = *(_QWORD **)(a1 + 16);
        v13 = &v12[2 * v11];
        do
        {
          if ( *v12 != -16 && *v12 != -8 )
          {
            v14 = v12[1];
            if ( (v14 & 4) != 0 )
            {
              v15 = (unsigned __int64 *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
              v16 = v15;
              if ( v15 )
              {
                if ( (unsigned __int64 *)*v15 != v15 + 2 )
                  _libc_free(*v15);
                j_j___libc_free_0(v16, 48);
              }
            }
          }
          v12 += 2;
        }
        while ( v13 != v12 );
      }
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      v17 = v19;
      ++*(_QWORD *)(a1 + 8);
      ++v18;
      *(_QWORD *)(a1 + 16) = v17;
      v19 = 0;
      *(_QWORD *)(a1 + 24) = v20;
      v20 = 0;
      *(_DWORD *)(a1 + 32) = v21;
      return j___libc_free_0(0);
    }
  }
  return result;
}
