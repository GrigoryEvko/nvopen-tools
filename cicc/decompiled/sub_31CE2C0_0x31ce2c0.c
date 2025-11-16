// Function: sub_31CE2C0
// Address: 0x31ce2c0
//
__int64 __fastcall sub_31CE2C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // ebx
  int v4; // edx
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v7; // r15
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  _QWORD *v12; // r13
  unsigned int v13; // edx
  unsigned int v14; // ecx
  __int64 result; // rax
  _WORD v16[24]; // [rsp-98h] [rbp-98h] BYREF
  __int64 v17; // [rsp-68h] [rbp-68h] BYREF
  __int64 v18; // [rsp-60h] [rbp-60h]
  __int64 v19; // [rsp-58h] [rbp-58h]
  __int64 v20; // [rsp-50h] [rbp-50h]
  __int64 v21; // [rsp-48h] [rbp-48h]
  __int64 v22; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a2 - 32);
  if ( !result || *(_BYTE *)result || *(_QWORD *)(result + 24) != *(_QWORD *)(a2 + 80) || !*(_DWORD *)(result + 36) )
    return result;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 )
    BUG();
  if ( *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v3 = *(_DWORD *)(v2 + 36);
  v4 = sub_BD2910(*(_QWORD *)(a1 + 312));
  if ( v3 <= 0x2014 )
  {
    if ( v3 > 0x200F )
    {
      result = 1;
    }
    else if ( v3 <= 0x1FF8 )
    {
      result = v3 < 0x1FF5 ? -1 : 1;
    }
    else
    {
      result = (unsigned int)-(v3 - 8199 > 1);
    }
    goto LABEL_13;
  }
  if ( v3 <= 0x2018 )
  {
    result = v3 < 0x2017 ? -1 : 1;
    if ( v4 != (_DWORD)result )
      return result;
    goto LABEL_24;
  }
  result = 0xFFFFFFFFLL;
  if ( v3 != 9056 )
  {
LABEL_13:
    if ( v4 != (_DWORD)result )
      return result;
LABEL_24:
    result = sub_CEA1F0(v3);
    if ( (_BYTE)result )
    {
      switch ( v3 )
      {
        case 0x2007u:
          v14 = 851968;
          return sub_31CD6C0(a1, a2, 0x22D5u, v14);
        case 0x2008u:
          v14 = 786432;
          return sub_31CD6C0(a1, a2, 0x22D5u, v14);
        case 0x2010u:
          v13 = 8915;
          goto LABEL_29;
        case 0x2011u:
          v13 = 8916;
          goto LABEL_29;
        case 0x2013u:
          v13 = 8917;
          goto LABEL_29;
        case 0x2014u:
          v13 = 8918;
LABEL_29:
          result = sub_31CD8A0(a1, a2, v13);
          break;
        default:
          return result;
      }
    }
    return result;
  }
  if ( v4 == 2 )
  {
    v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v5)) + 8LL);
    v18 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v5)) + 8LL);
    v6 = (__int64 *)sub_B43CA0(a2);
    v7 = sub_B6E160(v6, 0x2363u, (__int64)&v17, 2);
    v8 = *(_DWORD *)(a2 + 4);
    v16[16] = 257;
    v9 = v8 & 0x7FFFFFF;
    v17 = *(_QWORD *)(a2 - 32 * v9);
    v18 = *(_QWORD *)(a2 + 32 * (1 - v9));
    v19 = *(_QWORD *)(a2 + 32 * (2 - v9));
    v20 = *(_QWORD *)(a2 + 32 * (3 - v9));
    v10 = *(_QWORD *)(a1 + 304);
    v21 = *(_QWORD *)(a2 + 32 * (4 - v9));
    v11 = 0;
    v22 = *(_QWORD *)(v10 + 32 * (1LL - (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)));
    if ( v7 )
      v11 = *(_QWORD *)(v7 + 24);
    v12 = sub_BD2C40(88, 7u);
    if ( v12 )
    {
      sub_B44260((__int64)v12, **(_QWORD **)(v11 + 16), 56, 7u, a2 + 24, 0);
      v12[9] = 0;
      sub_B4A290((__int64)v12, v11, v7, &v17, 6, (__int64)v16, 0, 0);
    }
    sub_BD84D0(a2, (__int64)v12);
    return sub_B43D60((_QWORD *)a2);
  }
  return result;
}
