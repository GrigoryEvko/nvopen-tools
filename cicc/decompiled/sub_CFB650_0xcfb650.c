// Function: sub_CFB650
// Address: 0xcfb650
//
__int64 __fastcall sub_CFB650(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rsi
  __int64 v4; // rcx
  unsigned int v5; // edx
  _QWORD *v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rcx
  bool v11; // zf
  int v12; // edx
  int v13; // edx
  int v14; // r9d
  __int64 v15; // rcx
  _QWORD v16[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(v1 + 200);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 24);
    v4 = *(_QWORD *)(v1 + 184);
    v5 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v6 = (_QWORD *)(v4 + 48LL * v5);
    v7 = v6[3];
    if ( v3 == v7 )
    {
LABEL_3:
      result = v4 + 48 * result;
      if ( v6 != (_QWORD *)result )
      {
        v8 = v6[5];
        if ( v8 )
          sub_CFB3D0(v8, v3);
        v17 = -8192;
        v18 = 0;
        v16[0] = 2;
        result = v6[3];
        v16[1] = 0;
        if ( result == -8192 )
        {
          v6[4] = 0;
        }
        else
        {
          if ( !result || result == -4096 )
          {
            v6[3] = -8192;
            v13 = v17;
            v15 = v18;
            LOBYTE(result) = v17 != -4096;
            LOBYTE(v3) = v17 != -8192;
            LOBYTE(v13) = v17 != 0;
            result = v13 & (unsigned int)v3 & (unsigned int)result;
          }
          else
          {
            v9 = sub_BD60C0(v6 + 1);
            v10 = v17;
            v11 = v17 == -4096;
            v6[3] = v17;
            LOBYTE(v3) = v10 != 0;
            LOBYTE(v9) = v10 != -8192;
            v15 = v18;
            LOBYTE(v12) = !v11;
            result = (unsigned int)v3 & v12 & (unsigned int)v9;
          }
          v6[4] = v15;
          if ( (_BYTE)result )
            result = sub_BD60C0(v16);
        }
        --*(_DWORD *)(v1 + 192);
        ++*(_DWORD *)(v1 + 196);
      }
    }
    else
    {
      v14 = 1;
      while ( v7 != -4096 )
      {
        v5 = (result - 1) & (v14 + v5);
        v6 = (_QWORD *)(v4 + 48LL * v5);
        v7 = v6[3];
        if ( v3 == v7 )
          goto LABEL_3;
        ++v14;
      }
    }
  }
  return result;
}
