// Function: sub_13C6CB0
// Address: 0x13c6cb0
//
__int64 __fastcall sub_13C6CB0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // r13
  __int64 v3; // rcx
  unsigned int v4; // esi
  __int64 *v5; // rdx
  __int64 v6; // r8
  unsigned int v7; // edx
  __int64 (__fastcall *v8)(_QWORD *); // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  int v13; // r9d
  _QWORD v14[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+10h] [rbp-30h]
  __int64 v16; // [rsp+18h] [rbp-28h]

  result = *(_QWORD *)(a1 + 96);
  v2 = *(_QWORD **)(result - 24);
  if ( *(_QWORD **)(*(_QWORD *)(result - 32) + 16LL) != v2 )
  {
    while ( 1 )
    {
      v8 = *(__int64 (__fastcall **)(_QWORD *))(result - 16);
      *(_QWORD *)(result - 24) = v2 + 4;
      v9 = v2[2];
      v14[0] = 6;
      v14[1] = 0;
      v15 = v9;
      if ( v9 != 0 && v9 != -8 && v9 != -16 )
        sub_1649AC0(v14, *v2 & 0xFFFFFFFFFFFFFFF8LL);
      v16 = v2[3];
      v10 = v8(v14);
      if ( v15 != 0 && v15 != -8 && v15 != -16 )
        sub_1649B30(v14);
      v11 = *(unsigned int *)(a1 + 32);
      if ( !(_DWORD)v11 )
        goto LABEL_15;
      v3 = *(_QWORD *)(a1 + 16);
      v4 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( *v5 != v10 )
      {
        v12 = 1;
        while ( v6 != -8 )
        {
          v13 = v12 + 1;
          v4 = (v11 - 1) & (v12 + v4);
          v5 = (__int64 *)(v3 + 16LL * v4);
          v6 = *v5;
          if ( v10 == *v5 )
            goto LABEL_4;
          v12 = v13;
        }
        goto LABEL_15;
      }
LABEL_4:
      if ( v5 == (__int64 *)(v3 + 16 * v11) )
      {
LABEL_15:
        sub_13C69A0(a1, v10);
        result = *(_QWORD *)(a1 + 96);
        v2 = *(_QWORD **)(result - 24);
        if ( *(_QWORD **)(*(_QWORD *)(result - 32) + 16LL) == v2 )
          return result;
      }
      else
      {
        result = *(_QWORD *)(a1 + 96);
        v7 = *((_DWORD *)v5 + 2);
        if ( *(_DWORD *)(result - 8) > v7 )
        {
          *(_DWORD *)(result - 8) = v7;
          result = *(_QWORD *)(a1 + 96);
        }
        v2 = *(_QWORD **)(result - 24);
        if ( *(_QWORD **)(*(_QWORD *)(result - 32) + 16LL) == v2 )
          return result;
      }
    }
  }
  return result;
}
