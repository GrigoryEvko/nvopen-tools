// Function: sub_359F7A0
// Address: 0x359f7a0
//
void __fastcall sub_359F7A0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v4; // r15
  _BYTE *v5; // r12
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // eax
  _BYTE *v12; // r15
  __int64 v13; // rdi
  int v14; // r15d
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_BYTE **)(a2 + 32);
  v5 = &v4[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      v8 = (__int64)v4;
      if ( sub_2DADC00(v4) )
        break;
      v4 += 40;
      if ( v5 == v4 )
        return;
    }
    if ( v5 != v4 )
    {
      do
      {
        v11 = *(_DWORD *)(v8 + 8);
        if ( v11 < 0 )
        {
          v13 = a1[3];
          v19[0] = *(_DWORD *)(v8 + 8);
          v14 = sub_2EC06C0(
                  v13,
                  *(_QWORD *)(*(_QWORD *)(v13 + 56) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  byte_3F871B3,
                  0,
                  v9,
                  v10);
          sub_2EAB0C0(v8, v14);
          *sub_2FFAE70(a3, v19) = v14;
          if ( a4 )
            sub_359A420(a1, v19[0], v14, v15, v16, v17);
        }
        if ( (_BYTE *)(v8 + 40) == v5 )
          break;
        v12 = (_BYTE *)(v8 + 40);
        while ( 1 )
        {
          v8 = (__int64)v12;
          if ( sub_2DADC00(v12) )
            break;
          v12 += 40;
          if ( v5 == v12 )
            return;
        }
      }
      while ( v5 != v12 );
    }
  }
}
