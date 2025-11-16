// Function: sub_3171620
// Address: 0x3171620
//
char __fastcall sub_3171620(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v22; // [rsp+0h] [rbp-60h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, (__int64)a3, a4);
    v5 = *(_QWORD *)(a2 + 96);
    v23 = v5 + 40LL * *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, a2, v19, v20);
      v5 = *(_QWORD *)(a2 + 96);
    }
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 96);
    v23 = v5 + 40LL * *(_QWORD *)(a2 + 104);
  }
  v6 = v24;
  if ( v5 != v23 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v5 + 16);
      if ( v7 )
        break;
LABEL_25:
      v5 += 40;
      if ( v5 == v23 )
        return (char)v6;
    }
    while ( 1 )
    {
      v16 = *(_QWORD *)(v7 + 24);
      v17 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 80LL);
      if ( v17 )
        v17 -= 24;
      if ( *(_BYTE *)v16 != 84 )
        break;
      LODWORD(v6) = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
      if ( (unsigned int)v6 <= 1 )
      {
        v8 = *(_QWORD *)(v16 + 40);
LABEL_8:
        LOBYTE(v6) = sub_24F96E0(a3, v17, v8);
        if ( (_BYTE)v6 )
        {
          v24[0] = v5;
          v6 = (__int64 *)sub_31711D0(a1, v24, v9, v10, v11, v12);
          v15 = *((unsigned int *)v6 + 2);
          if ( v15 + 1 > (unsigned __int64)*((unsigned int *)v6 + 3) )
          {
            v22 = v6;
            sub_C8D5F0((__int64)v6, v6 + 2, v15 + 1, 8u, v13, v14);
            v6 = v22;
            v15 = *((unsigned int *)v22 + 2);
          }
          *(_QWORD *)(*v6 + 8 * v15) = v16;
          ++*((_DWORD *)v6 + 2);
        }
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_25;
    }
    v8 = *(_QWORD *)(v16 + 40);
    if ( *(_BYTE *)v16 == 85 )
    {
      v18 = *(_QWORD *)(v16 - 32);
      if ( v18 )
      {
        if ( !*(_BYTE *)v18
          && *(_QWORD *)(v18 + 24) == *(_QWORD *)(v16 + 80)
          && (*(_BYTE *)(v18 + 33) & 0x20) != 0
          && *(_DWORD *)(v18 + 36) == 62
          || !*(_BYTE *)v18
          && *(_QWORD *)(v18 + 24) == *(_QWORD *)(v16 + 80)
          && (*(_BYTE *)(v18 + 33) & 0x20) != 0
          && *(_DWORD *)(v18 + 36) == 61 )
        {
          v8 = sub_AA54C0(*(_QWORD *)(v16 + 40));
        }
      }
    }
    goto LABEL_8;
  }
  return (char)v6;
}
