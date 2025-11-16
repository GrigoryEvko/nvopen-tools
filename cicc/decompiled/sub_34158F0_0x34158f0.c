// Function: sub_34158F0
// Address: 0x34158f0
//
void __fastcall sub_34158F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  int v8; // ebx
  int v9; // r15d
  int v10; // edx
  int v11; // r8d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  int v18; // eax
  int v19; // ebx
  __int64 v20; // [rsp-78h] [rbp-78h] BYREF
  __int64 v21; // [rsp-70h] [rbp-70h] BYREF
  __int64 (__fastcall **v22)(); // [rsp-68h] [rbp-68h] BYREF
  __int64 v23; // [rsp-60h] [rbp-60h]
  __int64 v24; // [rsp-58h] [rbp-58h]
  __int64 *v25; // [rsp-50h] [rbp-50h]
  __int64 *v26; // [rsp-48h] [rbp-48h]

  if ( a2 != a3 )
  {
    v7 = a1;
    v8 = 0;
    v9 = *(_DWORD *)(a2 + 68);
    if ( v9 )
    {
      do
      {
        while ( !(unsigned __int8)sub_33CF8A0(a2, v8) )
        {
          if ( ++v8 == v9 )
            goto LABEL_7;
        }
        v10 = v8;
        v11 = v8++;
        sub_33F9B80(a1, a2, v10, a3, v11, 0, 0, 1);
      }
      while ( v8 != v9 );
    }
LABEL_7:
    sub_34151B0(a1, a2, a3, a4, a5, a6);
    v12 = *(_QWORD *)(a2 + 56);
    v13 = *(_QWORD *)(a1 + 768);
    *(_QWORD *)(a1 + 768) = &v22;
    v25 = &v20;
    v20 = v12;
    v21 = 0;
    v23 = v13;
    v24 = a1;
    v22 = off_4A36748;
    v26 = &v21;
    if ( v12 )
    {
      do
      {
        v14 = *(_QWORD *)(v12 + 16);
        sub_33EB970(a1, v14, v13);
        v15 = v20;
        do
        {
          v16 = *(_QWORD *)(v15 + 32);
          v20 = v16;
          if ( *(_QWORD *)v15 )
          {
            **(_QWORD **)(v15 + 24) = v16;
            if ( v16 )
              *(_QWORD *)(v16 + 24) = *(_QWORD *)(v15 + 24);
          }
          *(_QWORD *)v15 = a3;
          if ( a3 )
          {
            v17 = *(_QWORD *)(a3 + 56);
            *(_QWORD *)(v15 + 32) = v17;
            if ( v17 )
              *(_QWORD *)(v17 + 24) = v15 + 32;
            *(_QWORD *)(v15 + 24) = a3 + 56;
            *(_QWORD *)(a3 + 56) = v15;
          }
          if ( ((*(_BYTE *)(a2 + 32) & 4) != 0) != ((*(_BYTE *)(a3 + 32) & 4) != 0) )
            sub_33CEF80((_QWORD *)a1, v14);
          v15 = v20;
        }
        while ( v20 != v21 && v14 == *(_QWORD *)(v20 + 16) );
        sub_3415B20(a1, v14);
        v12 = v20;
      }
      while ( v21 != v20 );
      if ( a2 != *(_QWORD *)(a1 + 384) )
        goto LABEL_22;
      v19 = *(_DWORD *)(a1 + 392);
    }
    else
    {
      if ( a2 != *(_QWORD *)(a1 + 384) )
        goto LABEL_23;
      v18 = *(_DWORD *)(a1 + 392);
      v19 = v18;
      if ( !a3 )
      {
        *(_QWORD *)(a1 + 384) = 0;
        *(_DWORD *)(a1 + 392) = v18;
        goto LABEL_23;
      }
    }
    nullsub_1875();
    *(_QWORD *)(a1 + 384) = a3;
    *(_DWORD *)(a1 + 392) = v19;
    sub_33E2B60();
LABEL_22:
    v7 = v24;
    v13 = v23;
LABEL_23:
    *(_QWORD *)(v7 + 768) = v13;
  }
}
