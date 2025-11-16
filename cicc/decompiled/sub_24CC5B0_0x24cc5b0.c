// Function: sub_24CC5B0
// Address: 0x24cc5b0
//
void __fastcall sub_24CC5B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 *v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // [rsp-8h] [rbp-38h]
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_B33910(v16, (__int64 *)a1);
  v4 = v16[0];
  if ( v16[0] )
    goto LABEL_2;
  v5 = sub_B92180(a2);
  if ( v5 )
  {
    v6 = (__int64 *)(*(_QWORD *)(v5 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(v5 + 8) & 4) != 0 )
      v6 = (__int64 *)*v6;
    v7 = sub_B01860(v6, 0, 0, v5, 0, 0, 0, 1);
    sub_B10CB0(v16, (__int64)v7);
    v8 = v16[0];
    if ( v16[0] )
    {
      v9 = *(unsigned int *)(a1 + 8);
      v10 = *(_QWORD *)a1;
      v11 = *(_DWORD *)(a1 + 8);
      v12 = (_QWORD *)(*(_QWORD *)a1 + 16 * v9);
      if ( *(_QWORD **)a1 != v12 )
      {
        while ( *(_DWORD *)v10 )
        {
          v10 += 16;
          if ( v12 == (_QWORD *)v10 )
            goto LABEL_14;
        }
        *(_QWORD *)(v10 + 8) = v16[0];
        goto LABEL_13;
      }
LABEL_14:
      v13 = *(unsigned int *)(a1 + 12);
      if ( v9 >= v13 )
      {
        v14 = v9 + 1;
        if ( v13 < v14 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 0x10u, a1 + 16, v15);
          v12 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v12 = 0;
        v12[1] = v8;
        v8 = v16[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v12 )
        {
          *(_DWORD *)v12 = 0;
          v12[1] = v8;
          v11 = *(_DWORD *)(a1 + 8);
          v8 = v16[0];
        }
        *(_DWORD *)(a1 + 8) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v8 = v16[0];
    }
    if ( v8 )
    {
LABEL_13:
      v4 = v8;
LABEL_2:
      sub_B91220((__int64)v16, v4);
    }
  }
}
