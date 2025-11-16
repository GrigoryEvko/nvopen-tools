// Function: sub_2E513B0
// Address: 0x2e513b0
//
__int64 __fastcall sub_2E513B0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r15d
  __int64 v6; // r14
  int v7; // r15d
  int v8; // eax
  int v9; // r9d
  __int64 *v10; // r8
  unsigned int i; // ecx
  __int64 v12; // rdi
  __int64 *v13; // rbx
  __int64 v14; // r10
  char v15; // al
  char v16; // al
  __int64 *v17; // r8
  char v18; // al
  int v19; // [rsp+0h] [rbp-40h]
  int v20; // [rsp+0h] [rbp-40h]
  unsigned int v21; // [rsp+4h] [rbp-3Ch]
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  __int64 *v23; // [rsp+8h] [rbp-38h]
  __int64 *v24; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v8 = sub_2E8E920(a2);
    v9 = 1;
    v10 = 0;
    for ( i = v7 & v8; ; i = v7 & (v20 + v22) )
    {
      v12 = *a2;
      v13 = (__int64 *)(v6 + 16LL * i);
      v14 = *v13;
      if ( (unsigned __int64)(*v13 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v12 - 1) > 0xFFFFFFFFFFFFFFFDLL )
      {
        if ( v12 == v14 )
        {
LABEL_13:
          *a3 = v13;
          return 1;
        }
      }
      else
      {
        v19 = v9;
        v21 = i;
        v23 = v10;
        v15 = sub_2E88AF0(v12, *v13, 3);
        v10 = v23;
        i = v21;
        v9 = v19;
        if ( v15 )
          goto LABEL_13;
        v14 = *v13;
      }
      v20 = v9;
      v22 = i;
      v24 = v10;
      v16 = sub_2E4F140(v14, 0);
      v17 = v24;
      if ( v16 )
        break;
      v18 = sub_2E4F140(*v13, -1);
      v10 = v24;
      if ( !v24 && v18 )
        v10 = v13;
      v9 = v20 + 1;
    }
    if ( !v24 )
      v17 = v13;
    *a3 = v17;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
