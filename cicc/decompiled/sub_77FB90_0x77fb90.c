// Function: sub_77FB90
// Address: 0x77fb90
//
__int64 __fastcall sub_77FB90(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rcx
  _BYTE *v4; // r13
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // rdi
  _BYTE *v8; // r12
  _BYTE *v9; // rax
  _BYTE *v10; // rdx
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 48);
  v4 = *(_BYTE **)(a1 + 32);
  result = *(unsigned int *)(a1 + 4);
  if ( v2 <= 1 )
  {
    v8 = (_BYTE *)(a1 + 8);
    if ( (_DWORD)result && v8 != v4 )
    {
      v7 = 2;
      v6 = 2;
      goto LABEL_5;
    }
    v6 = 2;
  }
  else
  {
    v6 = v2 + (v2 >> 1) + 1;
    if ( (_DWORD)result )
    {
      result = a1 + 8;
      v7 = v2 + (v2 >> 1) + 1;
      if ( v4 != (_BYTE *)result )
        goto LABEL_5;
    }
    if ( v6 > 20 )
    {
      v7 = v2 + (v2 >> 1) + 1;
LABEL_5:
      v11 = v3;
      result = sub_823970(v7);
      v3 = v11;
      v8 = (_BYTE *)result;
      goto LABEL_6;
    }
    v8 = (_BYTE *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 4) = 1;
LABEL_6:
  if ( v8 != v4 )
  {
    v9 = v8;
    v10 = v4;
    if ( v3 > 0 )
    {
      do
      {
        if ( v9 )
          *v9 = *v10;
        ++v9;
        ++v10;
      }
      while ( v9 != &v8[v3] );
    }
    result = a1 + 8;
    if ( v4 == (_BYTE *)(a1 + 8) )
      *(_DWORD *)(a1 + 4) = 0;
    else
      result = sub_823A00(v4, v2);
  }
  *(_QWORD *)(a1 + 32) = v8;
  *(_QWORD *)(a1 + 40) = v6;
  return result;
}
