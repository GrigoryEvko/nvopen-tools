// Function: sub_2EC9FF0
// Address: 0x2ec9ff0
//
__int64 __fastcall sub_2EC9FF0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 **v6; // r12
  __int64 *v10; // rdx
  char v11; // zf
  __int64 v12; // rcx
  __int64 **i; // [rsp+8h] [rbp-78h]
  __int64 *v14; // [rsp+10h] [rbp-70h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  __int64 v16; // [rsp+18h] [rbp-68h]
  __int64 *v17; // [rsp+18h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  int v19; // [rsp+28h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int128 v21; // [rsp+38h] [rbp-48h]
  __int64 v22; // [rsp+48h] [rbp-38h]

  result = *(_QWORD *)(a2 + 72);
  v6 = *(__int64 ***)(a2 + 64);
  for ( i = (__int64 **)result; i != v6; ++v6 )
  {
    v10 = *v6;
    v11 = *(_DWORD *)(a2 + 24) == 1;
    v14 = a3;
    v18 = *a3;
    v16 = a4;
    v19 = *((_DWORD *)a3 + 2);
    v21 = 0u;
    v22 = 0;
    v20 = 0;
    sub_2EC9F20((__int64)a1, (__int64)&v18, v10, v11, a4, a4);
    v12 = 0;
    if ( *(_BYTE *)(a5 + 25) == BYTE1(v21) )
      v12 = a2;
    result = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64 *, __int64))(*a1 + 144))(a1, a5, &v18, v12);
    a4 = v16;
    a3 = v14;
    if ( (_BYTE)result )
    {
      if ( !v22 )
      {
        v15 = v16;
        v17 = a3;
        sub_2EC8FB0((__int64)&v18, a1[17], a1[2]);
        a4 = v15;
        a3 = v17;
      }
      *(_QWORD *)(a5 + 16) = v20;
      *(_WORD *)(a5 + 24) = v21;
      *(_QWORD *)(a5 + 26) = *(_QWORD *)((char *)&v21 + 2);
      *(_DWORD *)(a5 + 34) = *(_DWORD *)((char *)&v21 + 10);
      result = v22;
      *(_QWORD *)(a5 + 40) = v22;
    }
  }
  return result;
}
