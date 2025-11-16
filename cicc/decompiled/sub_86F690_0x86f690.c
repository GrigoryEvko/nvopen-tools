// Function: sub_86F690
// Address: 0x86f690
//
__int64 __fastcall sub_86F690(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  int v4; // edi
  __int64 *v5; // rbx
  __int64 *v6; // rax
  __int64 *v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rax
  char v10; // dl
  char v11; // [rsp-2Dh] [rbp-2Dh] BYREF
  int v12; // [rsp-2Ch] [rbp-2Ch] BYREF

  result = unk_4D03B90;
  if ( unk_4D03B90 >= 0 )
  {
    if ( a1 )
    {
      result = (__int64)&qword_4D03B98;
      v3 = qword_4D03B98 + 176LL * unk_4D03B90;
      if ( *(char *)(v3 + 5) < 0 )
      {
        result = (__int64)&dword_4F04C64;
        v4 = dword_4F04C64;
        if ( dword_4F04C64 >= 0 )
        {
          result = qword_4F04C68[0] + 776LL * dword_4F04C64;
          while ( *(_DWORD *)result != *(_DWORD *)(a1 + 40) )
          {
            --v4;
            result -= 776;
            if ( v4 == -1 )
              return result;
          }
          result = *(unsigned __int8 *)(result + 4);
          if ( (((_BYTE)result - 15) & 0xFD) == 0 || (_BYTE)result == 2 )
          {
            v5 = *(__int64 **)(v3 + 136);
            v6 = (__int64 *)*v5;
            if ( *v5 )
            {
              do
              {
                v7 = v6;
                v6 = (__int64 *)*v6;
              }
              while ( v6 );
              v5 = v7;
            }
            sub_7296F0(v4, &v12);
            *v5 = (__int64)sub_727640();
            sub_729730(v12);
            v8 = (_QWORD *)*v5;
            v9 = sub_87D510(a1, &v11);
            v10 = v11;
            v8[2] = v9;
            result = *v5;
            *(_BYTE *)(*v5 + 8) = v10;
          }
        }
      }
    }
  }
  return result;
}
