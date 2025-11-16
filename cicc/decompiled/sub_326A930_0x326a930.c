// Function: sub_326A930
// Address: 0x326a930
//
__int64 __fastcall sub_326A930(__int64 a1, unsigned int a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  int v4; // eax
  unsigned int v5; // r13d
  unsigned __int16 *v8; // rsi
  int v9; // eax
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 *v17; // rdi
  __int64 v18; // rcx
  int v19; // edx
  __int64 v20; // rdx
  __int16 v21; // [rsp+0h] [rbp-60h] BYREF
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int16 v23; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = a1;
  v4 = *(_DWORD *)(a1 + 24);
  LOBYTE(a1) = v4 == 35 || v4 == 11;
  if ( (_BYTE)a1 )
    return !(((*(_BYTE *)(v3 + 32) & 8) != 0) & a3);
  v5 = a1;
  if ( v4 == 156 || v4 == 168 )
  {
    v8 = (unsigned __int16 *)(*(_QWORD *)(v3 + 48) + 16LL * a2);
    v9 = *v8;
    v10 = *((_QWORD *)v8 + 1);
    v21 = v9;
    v22 = v10;
    if ( (_WORD)v9 )
    {
      if ( (unsigned __int16)(v9 - 17) > 0xD3u )
      {
        v23 = v9;
        v24 = v10;
        goto LABEL_21;
      }
      LOWORD(v9) = word_4456580[v9 - 1];
      v20 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v21) )
      {
        v24 = v10;
        v23 = 0;
LABEL_9:
        v25 = sub_3007260((__int64)&v23);
        LODWORD(v14) = v25;
        v26 = v15;
        goto LABEL_10;
      }
      LOWORD(v9) = sub_3009970((__int64)&v21, (__int64)v8, v11, v12, v13);
    }
    v23 = v9;
    v24 = v20;
    if ( !(_WORD)v9 )
      goto LABEL_9;
LABEL_21:
    if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v14 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
LABEL_10:
    v16 = *(__int64 **)(v3 + 40);
    v17 = &v16[5 * *(unsigned int *)(v3 + 64)];
    if ( v16 == v17 )
    {
      return 1;
    }
    else
    {
      while ( 1 )
      {
        v18 = *v16;
        v19 = *(_DWORD *)(*v16 + 24);
        if ( v19 != 51
          && (v19 != 11 && v19 != 35
           || (_DWORD)v14 != *(_DWORD *)(*(_QWORD *)(v18 + 96) + 32LL)
           || (*(_BYTE *)(v18 + 32) & 8) != 0 && a3) )
        {
          break;
        }
        v16 += 5;
        if ( v17 == v16 )
          return 1;
      }
    }
  }
  return v5;
}
