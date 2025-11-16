// Function: sub_29CFB10
// Address: 0x29cfb10
//
__int64 __fastcall sub_29CFB10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 *v12; // r12
  _BYTE *v13; // rcx
  __int64 result; // rax
  int v15; // eax
  int v16; // r9d
  __int64 v17; // [rsp+8h] [rbp-48h]
  _BYTE *v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 152);
  v8 = *(_QWORD *)(a1 + 136);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v7) )
      {
        v12 = v10 + 1;
        v13 = *(_BYTE **)(a1 + 624);
        v20 = *(_DWORD *)(a4 + 8);
        if ( v20 > 0x40 )
        {
          v18 = v13;
          sub_C43780((__int64)&v19, (const void **)a4);
          v13 = v18;
        }
        else
        {
          v19 = *(unsigned __int64 **)a4;
        }
        result = sub_29CF7D0(v12, a3, &v19, v13);
        if ( v20 > 0x40 )
        {
          if ( v19 )
          {
            v17 = result;
            j_j___libc_free_0_0((unsigned __int64)v19);
            return v17;
          }
        }
        return result;
      }
    }
    else
    {
      v15 = 1;
      while ( v11 != -4096 )
      {
        v16 = v15 + 1;
        v9 = (v7 - 1) & (v15 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_3;
        v15 = v16;
      }
    }
  }
  if ( sub_B2FC80(a2) || (unsigned __int8)sub_B2F6B0(a2) || (*(_BYTE *)(a2 + 80) & 2) != 0 )
    return 0;
  else
    return sub_9714E0(*(_QWORD *)(a2 - 32), a3, a4, *(_BYTE **)(a1 + 624));
}
