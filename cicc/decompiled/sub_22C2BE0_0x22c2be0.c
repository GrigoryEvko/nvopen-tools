// Function: sub_22C2BE0
// Address: 0x22c2be0
//
__int64 __fastcall sub_22C2BE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  unsigned int v4; // edx
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // r13
  __int64 v9; // rcx
  int v10; // ecx
  bool v11; // zf
  __int64 v12; // rax
  bool v13; // dl
  int v14; // edx
  int v15; // eax
  int v16; // edi
  _QWORD v17[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  char v19; // [rsp+20h] [rbp-60h]
  void *v20; // [rsp+30h] [rbp-50h]
  _QWORD v21[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v22; // [rsp+48h] [rbp-38h]
  unsigned __int8 v23; // [rsp+50h] [rbp-30h]

  v2 = a2;
  result = sub_22C1580(a1);
  if ( result )
  {
    v5 = result;
    v18 = a2;
    v17[0] = 2;
    v17[1] = 0;
    if ( a2 != -4096 && a2 != 0 && a2 != -8192 )
    {
      sub_BD73F0((__int64)v17);
      v2 = v18;
    }
    v19 = 0;
    v6 = *(_DWORD *)(v5 + 24);
    if ( v6 )
    {
      v4 = v6 - 1;
      v7 = *(_QWORD *)(v5 + 8);
      v6 = (v6 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v8 = v7 + 48LL * v6;
      v9 = *(_QWORD *)(v8 + 24);
      if ( v9 == v2 )
      {
LABEL_7:
        sub_22C1BD0((unsigned __int64 *)(v8 + 40));
        v23 = 0;
        v11 = *(_BYTE *)(v8 + 32) == 0;
        v21[0] = 2;
        v21[1] = 0;
        v20 = &unk_49DE8C0;
        v22 = -8192;
        if ( v11 )
        {
          result = *(_QWORD *)(v8 + 24);
          if ( result == -8192 )
          {
LABEL_15:
            --*(_DWORD *)(v5 + 16);
            ++*(_DWORD *)(v5 + 20);
            if ( v19 )
              return result;
            v14 = v18;
            LOBYTE(result) = v18 != 0;
            LOBYTE(v10) = v18 != -4096;
            LOBYTE(v14) = v18 != -8192;
            result = v14 & v10 & (unsigned int)result;
            goto LABEL_17;
          }
          if ( result && result != -4096 )
          {
            sub_BD60C0((_QWORD *)(v8 + 8));
            v12 = v22;
            v13 = v22 != -8192 && v22 != -4096 && v22 != 0;
            goto LABEL_12;
          }
          *(_QWORD *)(v8 + 24) = -8192;
        }
        else
        {
          *(_QWORD *)(v8 + 24) = 0;
          v12 = v22;
          if ( v22 )
          {
            v13 = v22 != -8192 && v22 != -4096;
LABEL_12:
            *(_QWORD *)(v8 + 24) = v12;
            if ( v13 )
              sub_BD6050((unsigned __int64 *)(v8 + 8), v21[0] & 0xFFFFFFFFFFFFFFF8LL);
          }
        }
        result = v23;
        *(_BYTE *)(v8 + 32) = v23;
        if ( !(_BYTE)result )
        {
          result = v22;
          v20 = &unk_49DB368;
          if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            result = sub_BD60C0(v21);
        }
        goto LABEL_15;
      }
      v16 = 1;
      while ( v9 != -4096 )
      {
        v6 = v4 & (v16 + v6);
        v8 = v7 + 48LL * v6;
        v9 = *(_QWORD *)(v8 + 24);
        if ( v9 == v2 )
          goto LABEL_7;
        ++v16;
      }
    }
    LOBYTE(v6) = v2 != -4096;
    LOBYTE(v4) = v2 != 0;
    v15 = v4 & v6;
    LOBYTE(v4) = v2 != -8192;
    result = v4 & v15;
LABEL_17:
    if ( (_BYTE)result )
      return sub_BD60C0(v17);
  }
  return result;
}
