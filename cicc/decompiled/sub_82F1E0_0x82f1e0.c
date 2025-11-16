// Function: sub_82F1E0
// Address: 0x82f1e0
//
__int64 __fastcall sub_82F1E0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r13
  _QWORD *v8; // rdx
  __int64 v9; // r15
  __int64 *v10; // r14
  __int64 *v11; // rsi
  __int64 *v12; // r13
  int v13; // eax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 *v20; // [rsp+18h] [rbp-48h]
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  *(_BYTE *)(a3 + 18) |= 1u;
  result = (2 * (a2 & 1)) | *(_BYTE *)(a1 + 18) & 0xFDu;
  *(_BYTE *)(a1 + 18) = (2 * (a2 & 1)) | *(_BYTE *)(a1 + 18) & 0xFD;
  if ( (*(_BYTE *)(a3 + 18) & 4) != 0 && *(_BYTE *)(a3 + 16) == 1 )
  {
    if ( a2 )
    {
      result = *(unsigned __int8 *)(a1 + 16);
      if ( (_BYTE)result == 2 )
      {
        result = sub_8D6D50(a1 + 144);
        v7 = result;
      }
      else
      {
        if ( (_BYTE)result != 1 )
          return result;
        result = sub_8D7160(*(_QWORD *)(a1 + 144), 1);
        v7 = result;
      }
    }
    else
    {
      result = sub_82F150((_BYTE *)a1);
      v7 = result;
    }
    if ( v7 )
    {
      result = (__int64)&dword_4F04C44;
      if ( dword_4F04C44 == -1 )
      {
        v8 = qword_4F04C68;
        result = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(result + 6) & 6) == 0 && *(_BYTE *)(result + 4) != 12 )
        {
          while ( *(_BYTE *)(v7 + 140) == 12 )
            v7 = *(_QWORD *)(v7 + 160);
          v9 = *(_QWORD *)(a3 + 144);
          v10 = *(__int64 **)(v9 + 56);
          result = v10[5];
          v11 = *(__int64 **)(result + 32);
          if ( v11 == (__int64 *)v7
            || (result = sub_8D97D0(v7, v11, 0, v5, v6), (_DWORD)result)
            || (v11 = *(__int64 **)(a1 + 144),
                v20 = v11,
                result = sub_6EC3E0((__int64)v10, (__int64)v11, v7, v5, v6),
                v12 = (__int64 *)result,
                v10 == (__int64 *)result)
            || result && dword_4F07588 && (result = v10[4], v12[4] == result) && result )
          {
            *(_BYTE *)(a3 + 18) &= ~4u;
          }
          else
          {
            v19 = *(_QWORD *)(v12[5] + 32);
            v18 = *v11;
            v13 = sub_8D2E30(*v11);
            v14 = v18;
            if ( v13 )
              v14 = sub_8D46C0(v18);
            v11 = (__int64 *)v19;
            result = sub_6EC0E0(v20, v19, v14, v21);
            if ( result )
            {
              v5 = (__int64)v20;
              *(_QWORD *)(result + 72) = v20;
              v15 = (_QWORD *)v21[0];
              *(_QWORD *)(a1 + 144) = v21[0];
              *(_QWORD *)a1 = *v15;
              v16 = *(_QWORD *)(a3 + 88);
              if ( v16 )
              {
                v8 = (_QWORD *)*v12;
                *(_QWORD *)(v16 + 16) = *v12;
              }
              *(_QWORD *)(v9 + 56) = v12;
              *(_QWORD *)(a3 + 8) = v10[19];
              v17 = (_QWORD *)v12[19];
              *(_QWORD *)v9 = v17;
              if ( (*(_BYTE *)(v9 + 25) & 3) == 0 )
              {
                v11 = 0;
                *(_QWORD *)v9 = sub_72D2E0(v17);
              }
              result = *(_QWORD *)v9;
              v10 = v12;
              *(_BYTE *)(a3 + 18) &= ~4u;
              *(_QWORD *)a3 = result;
            }
          }
          if ( (*(_BYTE *)(a3 + 18) & 4) == 0 )
            return (__int64)sub_6E1D20(v10, (__int64)v11, (__int64)v8, v5, v6);
        }
      }
    }
  }
  return result;
}
