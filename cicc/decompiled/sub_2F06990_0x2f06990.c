// Function: sub_2F06990
// Address: 0x2f06990
//
__int64 __fastcall sub_2F06990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r14
  __int64 v15; // r15
  unsigned __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  __int64 v23; // [rsp+28h] [rbp-88h]
  _QWORD v24[4]; // [rsp+38h] [rbp-78h] BYREF
  __int64 v25; // [rsp+58h] [rbp-58h]
  _QWORD v26[10]; // [rsp+60h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 16);
  v10 = *(_QWORD *)(v8 + 16);
  v25 = *(_QWORD *)a3;
  v24[1] = v9;
  v24[2] = v10;
  v21 = v10;
  v24[3] = v26;
  v22 = v25;
  v26[0] = 0;
  if ( sub_2F063B0(a1 + 8, v9, a3, v25, a5, a6, v9, v10, v26, v25) )
  {
    v12 = *(_QWORD *)(a3 + 40);
    v13 = 16LL * *(unsigned int *)(a3 + 48);
    v14 = v12 + v13;
    if ( v12 + v13 != v12 )
    {
      v15 = *(_QWORD *)(a3 + 40);
      while ( 1 )
      {
        v20 = (*(__int64 *)v15 >> 1) & 3;
        if ( v20 != 3 )
          break;
        if ( *(_DWORD *)(v15 + 8) <= 3u )
        {
LABEL_6:
          v16 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
          v17 = v16;
          if ( *(_DWORD *)(v16 + 200) != -1 )
          {
            v23 = *(_QWORD *)v16;
            if ( (unsigned __int8)sub_2F06550(*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL, 2u, v16, v12, v13, v11) )
            {
              v24[0] = v23;
              v26[0] = v9;
              v26[1] = v21;
              v26[2] = v24;
              v26[3] = v22;
              if ( sub_2F063B0(a1 + 8, 2, v18, v12, v13, v11, v9, v21, v24, v22) )
              {
                result = sub_2F065C0(a2, v17, a3);
                if ( (_BYTE)result )
                  return result;
              }
            }
          }
LABEL_10:
          v15 += 16;
          if ( v14 == v15 )
            return 0;
        }
        else
        {
          v15 += 16;
          if ( v14 == v15 )
            return 0;
        }
      }
      if ( v20 == 1 || v20 == 2 )
        goto LABEL_10;
      goto LABEL_6;
    }
  }
  return 0;
}
