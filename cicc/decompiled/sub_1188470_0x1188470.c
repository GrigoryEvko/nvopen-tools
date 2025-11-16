// Function: sub_1188470
// Address: 0x1188470
//
__int64 __fastcall sub_1188470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 v11; // r8
  __int64 v12; // r10
  char v13; // al
  __int64 v14; // rax
  _QWORD *v15; // r11
  unsigned int v16; // r12d
  _BYTE *v17; // r9
  int v18; // eax
  _QWORD *v19; // r11
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp-10h] [rbp-80h]
  __int64 v26; // [rsp+0h] [rbp-70h]
  _QWORD *v27; // [rsp+8h] [rbp-68h]
  unsigned __int8 v28; // [rsp+17h] [rbp-59h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  _QWORD *v33; // [rsp+20h] [rbp-50h]
  _QWORD *v34; // [rsp+28h] [rbp-48h]
  __int64 v35[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a5 != 2
    && *(_BYTE *)a2 > 0x1Cu
    && (v5 = *(_QWORD *)(a2 + 16)) != 0
    && !*(_QWORD *)(v5 + 8)
    && (v28 = sub_991A70((unsigned __int8 *)a2, 0, 0, 0, 0, 0, 0)) != 0
    && ((v12 = a1, (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 > 1)
     || (v13 = sub_98C610((char *)a2, 0, v25), v12 = a1, v13)) )
  {
    v14 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v15 = *(_QWORD **)(a2 - 8);
      v34 = &v15[v14];
    }
    else
    {
      v34 = (_QWORD *)a2;
      v15 = (_QWORD *)(a2 - v14 * 8);
    }
    v6 = 0;
    if ( v34 != v15 )
    {
      v16 = a5 + 1;
      do
      {
        v17 = (_BYTE *)*v15;
        if ( a3 == *v15 )
        {
          v20 = (__int64 *)v15[2];
          v21 = v15[1];
          *v20 = v21;
          if ( v21 )
          {
            v20 = (__int64 *)v15[2];
            *(_QWORD *)(v21 + 16) = v20;
          }
          *v15 = a4;
          if ( a4 )
          {
            v21 = *(_QWORD *)(a4 + 16);
            v20 = (__int64 *)(a4 + 16);
            v15[1] = v21;
            if ( v21 )
              *(_QWORD *)(v21 + 16) = v15 + 1;
            v15[2] = v20;
            *(_QWORD *)(a4 + 16) = v15;
          }
          v22 = *(_QWORD *)(v12 + 40);
          if ( *v17 > 0x1Cu )
          {
            v26 = v12;
            v27 = v15;
            v35[0] = (__int64)v17;
            v30 = (__int64)v17;
            sub_1187E30(v22 + 2096, v35, v21, (__int64)v20, v11, (__int64)v17);
            v17 = (_BYTE *)v30;
            v23 = v22 + 2096;
            v15 = v27;
            v12 = v26;
            v24 = *(_QWORD *)(v30 + 16);
            if ( !v24 || *(_QWORD *)(v24 + 8) )
            {
              v22 = *(_QWORD *)(v26 + 40);
            }
            else
            {
              v35[0] = *(_QWORD *)(v24 + 24);
              sub_1187E30(v23, v35, v21, (__int64)v20, v11, v30);
              v12 = v26;
              v15 = v27;
              v22 = *(_QWORD *)(v26 + 40);
            }
          }
          v31 = v12;
          v33 = v15;
          v35[0] = a2;
          sub_1187E30(v22 + 2096, v35, v21, (__int64)v20, v11, (__int64)v17);
          v6 = v28;
          v19 = v33;
          v12 = v31;
        }
        else
        {
          v29 = v15;
          v32 = v12;
          v18 = sub_1188470(v12, *v15, a3, a4, v16);
          v19 = v29;
          v12 = v32;
          v6 |= v18;
        }
        v15 = v19 + 4;
      }
      while ( v34 != v15 );
    }
  }
  else
  {
    return 0;
  }
  return v6;
}
