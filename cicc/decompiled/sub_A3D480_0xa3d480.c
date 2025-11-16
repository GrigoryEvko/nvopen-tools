// Function: sub_A3D480
// Address: 0xa3d480
//
__int64 __fastcall sub_A3D480(char *a1, char *a2, char *a3, char *a4, __int64 a5, __int64 a6)
{
  char *v7; // r12
  char *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  signed __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a3;
  v8 = a1;
  v21[0] = a6;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      if ( sub_A3D0E0((__int64)v21, v7, v8) )
      {
        v9 = *(_QWORD *)v7;
        a5 += 16;
        v7 += 16;
        *(_QWORD *)(a5 - 16) = v9;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)v7 - 2);
        if ( v8 == a2 )
          break;
      }
      else
      {
        v10 = *(_QWORD *)v8;
        v8 += 16;
        a5 += 16;
        *(_QWORD *)(a5 - 16) = v10;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)v8 - 2);
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  v11 = a2 - v8;
  v12 = (a2 - v8) >> 4;
  if ( a2 - v8 > 0 )
  {
    v13 = a5;
    do
    {
      v14 = *(_QWORD *)v8;
      v13 += 16;
      v8 += 16;
      *(_QWORD *)(v13 - 16) = v14;
      *(_DWORD *)(v13 - 8) = *((_DWORD *)v8 - 2);
      --v12;
    }
    while ( v12 );
    a5 += v11;
  }
  v15 = a4 - v7;
  v16 = (a4 - v7) >> 4;
  if ( a4 - v7 > 0 )
  {
    v17 = a5;
    do
    {
      v18 = *(_QWORD *)v7;
      v17 += 16;
      v7 += 16;
      *(_QWORD *)(v17 - 16) = v18;
      *(_DWORD *)(v17 - 8) = *((_DWORD *)v7 - 2);
      --v16;
    }
    while ( v16 );
    if ( v15 <= 0 )
      v15 = 16;
    a5 += v15;
  }
  return a5;
}
