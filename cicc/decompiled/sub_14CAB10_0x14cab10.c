// Function: sub_14CAB10
// Address: 0x14cab10
//
__int64 __fastcall sub_14CAB10(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rsi
  __int64 v4; // rcx
  unsigned int v5; // edx
  _QWORD *v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdx
  bool v10; // zf
  int v11; // r9d
  _QWORD v12[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(v1 + 184);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 24);
    v4 = *(_QWORD *)(v1 + 168);
    v5 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v6 = (_QWORD *)(v4 + 48LL * v5);
    v7 = v6[3];
    if ( v3 == v7 )
    {
LABEL_4:
      result = v4 + 48 * result;
      if ( v6 != (_QWORD *)result )
      {
        v8 = v6[5];
        if ( v8 )
          sub_14CA8A0(v8);
        v13 = -16;
        v14 = 0;
        v12[0] = 2;
        result = v6[3];
        v12[1] = 0;
        if ( result == -16 )
        {
          v6[4] = 0;
        }
        else if ( !result || result == -8 )
        {
          v6[3] = -16;
          v9 = v13;
          v10 = v13 == 0;
          v6[4] = v14;
          LOBYTE(result) = !v10;
          LOBYTE(v3) = v9 != -8;
          LOBYTE(v9) = v9 != -16;
          result = (unsigned int)v9 & (unsigned int)v3 & (unsigned int)result;
          if ( (_BYTE)result )
            result = sub_1649B30(v12);
        }
        else
        {
          sub_1649B30(v6 + 1);
          v6[3] = v13;
          result = v14;
          v6[4] = v14;
        }
        --*(_DWORD *)(v1 + 176);
        ++*(_DWORD *)(v1 + 180);
      }
    }
    else
    {
      v11 = 1;
      while ( v7 != -8 )
      {
        v5 = (result - 1) & (v11 + v5);
        v6 = (_QWORD *)(v4 + 48LL * v5);
        v7 = v6[3];
        if ( v3 == v7 )
          goto LABEL_4;
        ++v11;
      }
    }
  }
  return result;
}
