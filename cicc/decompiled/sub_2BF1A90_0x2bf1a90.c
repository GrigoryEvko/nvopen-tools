// Function: sub_2BF1A90
// Address: 0x2bf1a90
//
void __fastcall sub_2BF1A90(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // r9
  __int64 v5; // r12
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // ecx
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 v19[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_B10CD0(a2);
  if ( !v2
    || !(unsigned __int8)sub_B921D0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 904) + 48LL) + 72LL))
    || LOBYTE(qword_4F813A8[8]) )
  {
    v3 = *(_QWORD *)(a1 + 904);
    sub_B10CB0(v19, v2);
    v5 = v19[0];
    if ( v19[0] )
    {
      v6 = *(unsigned int *)(v3 + 8);
      v7 = *(_QWORD **)v3;
      v8 = *(_DWORD *)(v3 + 8);
      v9 = (_QWORD *)(*(_QWORD *)v3 + 16 * v6);
      if ( *(_QWORD **)v3 != v9 )
      {
        while ( *(_DWORD *)v7 )
        {
          v7 += 2;
          if ( v9 == v7 )
            goto LABEL_15;
        }
        v7[1] = v19[0];
        goto LABEL_9;
      }
LABEL_15:
      v16 = *(unsigned int *)(v3 + 12);
      if ( v6 >= v16 )
      {
        v17 = v6 + 1;
        if ( v16 < v17 )
        {
          sub_C8D5F0(v3, (const void *)(v3 + 16), v17, 0x10u, v3 + 16, v4);
          v9 = (_QWORD *)(*(_QWORD *)v3 + 16LL * *(unsigned int *)(v3 + 8));
        }
        *v9 = 0;
        v9[1] = v5;
        ++*(_DWORD *)(v3 + 8);
        v5 = v19[0];
      }
      else
      {
        if ( v9 )
        {
          *(_DWORD *)v9 = 0;
          v9[1] = v5;
          v8 = *(_DWORD *)(v3 + 8);
          v5 = v19[0];
        }
        *(_DWORD *)(v3 + 8) = v8 + 1;
      }
    }
    else
    {
      sub_93FB40(v3, 0);
      v5 = v19[0];
    }
    if ( !v5 )
      return;
LABEL_9:
    sub_B91220((__int64)v19, v5);
    return;
  }
  v10 = sub_2A114C0(v2, *(_DWORD *)(a1 + 8) * **(_DWORD **)(*(_QWORD *)(a1 + 920) + 144LL));
  v19[1] = v11;
  v19[0] = (__int64)v10;
  if ( (_BYTE)v11 )
  {
    v12 = *(_QWORD *)(a1 + 904);
    sub_B10CB0(&v18, v19[0]);
    sub_F80810(v12, 0, v18, v13, v14, v15);
    if ( v18 )
      sub_B91220((__int64)&v18, v18);
  }
}
