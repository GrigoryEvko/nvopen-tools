// Function: sub_D129F0
// Address: 0xd129f0
//
__int64 __fastcall sub_D129F0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  unsigned int v9; // edx
  __int64 (__fastcall *v10)(unsigned __int64 *); // r14
  __int64 v11; // rax
  int v12; // edx
  int v13; // r9d
  unsigned __int64 v14[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h]
  char v16; // [rsp+18h] [rbp-38h]
  __int64 v17; // [rsp+20h] [rbp-30h]

  result = *(_QWORD *)(a1 + 96);
  v2 = *(_QWORD *)(result - 24);
  if ( *(_QWORD *)(*(_QWORD *)(result - 32) + 24LL) != v2 )
  {
    while ( 1 )
    {
      v10 = *(__int64 (__fastcall **)(unsigned __int64 *))(result - 16);
      *(_QWORD *)(result - 24) = v2 + 40;
      v16 = 0;
      if ( *(_BYTE *)(v2 + 24) )
      {
        v11 = *(_QWORD *)(v2 + 16);
        v14[0] = 6;
        v14[1] = 0;
        v15 = v11;
        if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          sub_BD6050(v14, *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
        v16 = 1;
      }
      v17 = *(_QWORD *)(v2 + 32);
      v3 = v10(v14);
      if ( v16 && (v16 = 0, v15 != 0 && v15 != -4096) && v15 != -8192 )
      {
        sub_BD60C0(v14);
        v4 = *(unsigned int *)(a1 + 32);
        v5 = *(_QWORD *)(a1 + 16);
        if ( !(_DWORD)v4 )
          goto LABEL_18;
      }
      else
      {
        v4 = *(unsigned int *)(a1 + 32);
        v5 = *(_QWORD *)(a1 + 16);
        if ( !(_DWORD)v4 )
          goto LABEL_18;
      }
      v6 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v3 != *v7 )
      {
        v12 = 1;
        while ( v8 != -4096 )
        {
          v13 = v12 + 1;
          v6 = (v4 - 1) & (v12 + v6);
          v7 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v7;
          if ( v3 == *v7 )
            goto LABEL_6;
          v12 = v13;
        }
        goto LABEL_18;
      }
LABEL_6:
      if ( v7 == (__int64 *)(v5 + 16 * v4) )
      {
LABEL_18:
        sub_D126D0(a1, v3);
        result = *(_QWORD *)(a1 + 96);
        v2 = *(_QWORD *)(result - 24);
        if ( *(_QWORD *)(*(_QWORD *)(result - 32) + 24LL) == v2 )
          return result;
      }
      else
      {
        result = *(_QWORD *)(a1 + 96);
        v9 = *((_DWORD *)v7 + 2);
        if ( *(_DWORD *)(result - 8) > v9 )
        {
          *(_DWORD *)(result - 8) = v9;
          result = *(_QWORD *)(a1 + 96);
        }
        v2 = *(_QWORD *)(result - 24);
        if ( *(_QWORD *)(*(_QWORD *)(result - 32) + 24LL) == v2 )
          return result;
      }
    }
  }
  return result;
}
