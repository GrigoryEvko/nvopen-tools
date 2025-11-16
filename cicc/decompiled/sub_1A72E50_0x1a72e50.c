// Function: sub_1A72E50
// Address: 0x1a72e50
//
unsigned __int64 __fastcall sub_1A72E50(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  _QWORD *v4; // r15
  int v5; // r14d
  unsigned int i; // ebx
  unsigned int v7; // esi
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // edx
  int v11; // ecx
  __int64 v12; // r8
  unsigned int v13; // edx
  _QWORD *v14; // rsi
  _QWORD *v15; // rdi
  int v16; // esi
  int v17; // r9d

  result = sub_157EBA0(a2);
  if ( result )
  {
    v4 = (_QWORD *)result;
    v5 = sub_15F4D60(result);
    if ( v5 )
    {
      for ( i = 0; i != v5; ++i )
      {
        v7 = i;
        v8 = sub_15F4DF0((__int64)v4, v7);
        sub_1A72700(a1, a2, v8);
      }
    }
    v9 = *(_QWORD *)(a1 + 208);
    if ( v9 )
    {
      v10 = *(_DWORD *)(v9 + 184);
      if ( v10 )
      {
        v11 = v10 - 1;
        v12 = *(_QWORD *)(v9 + 168);
        v13 = (v10 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v14 = (_QWORD *)(v12 + 8LL * v13);
        v15 = (_QWORD *)*v14;
        if ( v4 == (_QWORD *)*v14 )
        {
LABEL_8:
          *v14 = -16;
          --*(_DWORD *)(v9 + 176);
          ++*(_DWORD *)(v9 + 180);
        }
        else
        {
          v16 = 1;
          while ( v15 != (_QWORD *)-8LL )
          {
            v17 = v16 + 1;
            v13 = v11 & (v16 + v13);
            v14 = (_QWORD *)(v12 + 8LL * v13);
            v15 = (_QWORD *)*v14;
            if ( v4 == (_QWORD *)*v14 )
              goto LABEL_8;
            v16 = v17;
          }
        }
      }
    }
    return sub_15F20C0(v4);
  }
  return result;
}
