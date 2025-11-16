// Function: sub_657CE0
// Address: 0x657ce0
//
__int64 __fastcall sub_657CE0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rcx
  __int64 result; // rax
  char v4; // dl
  __int64 v5; // rax
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r12
  _QWORD *i; // rax
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 j; // r12
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  __int64 *v18[5]; // [rsp+18h] [rbp-28h] BYREF

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 120LL);
  if ( (unsigned int)sub_8D2FB0(v1) )
    v1 = sub_8D46C0(v1);
  v2 = *(_QWORD *)(a1 + 120);
  result = *(_QWORD *)&dword_4D03B80;
  if ( v2 != *(_QWORD *)&dword_4D03B80 )
  {
    if ( *(_BYTE *)(a1 + 177) == 5 )
    {
      if ( (unsigned int)sub_8D3410(v1) )
      {
        return *(_QWORD *)(a1 + 120);
      }
      else
      {
        v11 = 1;
        for ( i = *(_QWORD **)(*(_QWORD *)(a1 + 128) + 128LL); a1 != i[2]; ++v11 )
          i = (_QWORD *)*i;
        if ( (unsigned int)sub_643950(v1, &v17, v18, 1, (__int64)dword_4F07508) )
        {
          v13 = 0;
          if ( (*(_BYTE *)(v1 + 140) & 0xFB) == 8 )
            v13 = (unsigned int)sub_8D4C10(v1, dword_4F077C4 != 2);
          v14 = v18[0];
          for ( j = v11 - 1; j; --j )
          {
            do
              v14 = (__int64 *)v14[14];
            while ( v14 && (!*v14 || !v14[1] && (v14[18] & 4) != 0) );
            v18[0] = v14;
          }
          if ( (v14[18] & 0x20) != 0 )
            v13 = (unsigned int)v13 & 0xFFFFFFFE;
          return sub_73C570(v14[15], v13, -1);
        }
        else
        {
          return sub_72C930(v1);
        }
      }
    }
    else
    {
      v4 = *(_BYTE *)(v2 + 140);
      if ( v4 == 12 )
      {
        v5 = *(_QWORD *)(a1 + 120);
        do
        {
          v5 = *(_QWORD *)(v5 + 160);
          v4 = *(_BYTE *)(v5 + 140);
        }
        while ( v4 == 12 );
      }
      result = *(_QWORD *)(a1 + 120);
      if ( v4 )
      {
        v6 = *(_QWORD *)(a1 + 128);
        v7 = *(_QWORD **)(v6 + 128);
        if ( a1 == v7[2] )
        {
          v9 = 0;
        }
        else
        {
          v8 = 1;
          do
          {
            v7 = (_QWORD *)*v7;
            v9 = v8++;
          }
          while ( a1 != v7[2] );
        }
        v10 = *(_QWORD *)(v6 + 120);
        v18[0] = 0;
        if ( (unsigned int)sub_8D2FB0(v10) )
          v10 = sub_8D46C0(v10);
        v16 = sub_643630(v6, v10, v9, 1, (__int64)dword_4F07508, (__int64)v18);
        sub_6E1990(v18[0]);
        return v16;
      }
    }
  }
  return result;
}
