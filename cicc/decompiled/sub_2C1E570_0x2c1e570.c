// Function: sub_2C1E570
// Address: 0x2c1e570
//
__int64 *__fastcall sub_2C1E570(__int64 a1, __int64 a2)
{
  char v3; // r15
  int v4; // ebx
  __int64 v5; // r12
  int v6; // ecx
  __int64 v7; // rax
  unsigned int **v8; // rdi
  __int64 v10; // [rsp+0h] [rbp-80h]
  __int64 v11; // [rsp+10h] [rbp-70h]
  int v12; // [rsp+18h] [rbp-68h]
  unsigned int v13; // [rsp+1Ch] [rbp-64h]
  __int64 v14[4]; // [rsp+20h] [rbp-60h] BYREF
  char v15; // [rsp+40h] [rbp-40h]
  char v16; // [rsp+41h] [rbp-3Fh]

  v14[0] = *(_QWORD *)(a1 + 88);
  if ( v14[0] )
    sub_2AAAFA0(v14);
  sub_2BF1A90(a2, (__int64)v14);
  sub_9C6650(v14);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 56) + 1) >> 1;
  v10 = a1 + 96;
  v3 = sub_2C46C30(a1 + 96);
  if ( v13 )
  {
    v4 = 0;
    v5 = 0;
    do
    {
      while ( !v4 )
      {
        v4 = 1;
        v5 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), v3);
        if ( v13 == 1 )
          return sub_2BF26E0(a2, v10, v5, v3);
      }
      v6 = 2 * v4++;
      v12 = v6;
      v11 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (v6 - (*(_DWORD *)(a1 + 56) & 1u))), v3);
      v7 = sub_2BFB640(
             a2,
             *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * ((unsigned int)((*(_DWORD *)(a1 + 56) & 1) == 0) + v12)),
             v3);
      v8 = *(unsigned int ***)(a2 + 904);
      v16 = 1;
      v15 = 3;
      v14[0] = (__int64)"predphi";
      v5 = sub_B36550(v8, v7, v11, v5, (__int64)v14, 0);
    }
    while ( v13 != v4 );
  }
  else
  {
    v5 = 0;
  }
  return sub_2BF26E0(a2, v10, v5, v3);
}
