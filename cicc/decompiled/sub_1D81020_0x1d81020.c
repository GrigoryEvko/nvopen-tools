// Function: sub_1D81020
// Address: 0x1d81020
//
__int64 __fastcall sub_1D81020(_QWORD *a1)
{
  __int64 v1; // r10
  __int64 v2; // r13
  __int64 v3; // r14
  _QWORD *v4; // r15
  char v5; // bl
  _QWORD *v6; // rax
  _QWORD *v7; // r9
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v14; // [rsp+10h] [rbp-70h]
  _QWORD *v15; // [rsp+18h] [rbp-68h]
  unsigned int v16; // [rsp+2Ch] [rbp-54h] BYREF
  const char *v17; // [rsp+30h] [rbp-50h] BYREF
  char v18; // [rsp+40h] [rbp-40h]
  char v19; // [rsp+41h] [rbp-3Fh]

  v1 = *(a1 - 3);
  if ( *(_BYTE *)(v1 + 16) == 87 )
  {
    if ( *(_DWORD *)(v1 + 64) == 1
      && **(_DWORD **)(v1 + 56) == 1
      && (v3 = *(_QWORD *)(v1 - 48), *(_BYTE *)(v3 + 16) == 87) )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v3 - 48) + 16LL) == 9 && *(_DWORD *)(v3 + 64) == 1 && !**(_DWORD **)(v3 + 56) )
      {
        v2 = *(_QWORD *)(v1 - 24);
        v8 = *(_QWORD *)(v3 - 24);
        if ( *(_BYTE *)(v2 + 16) != 54 )
          v2 = 0;
        v4 = (_QWORD *)*(a1 - 3);
        v5 = 1;
        if ( v8 )
        {
          v15 = (_QWORD *)*(a1 - 3);
          sub_15F20C0(a1);
          v4 = v15;
LABEL_18:
          if ( !v4[1] )
            sub_15F20C0(v4);
          if ( !*(_QWORD *)(v3 + 8) )
            sub_15F20C0((_QWORD *)v3);
          if ( v2 && !*(_QWORD *)(v2 + 8) )
            sub_15F20C0((_QWORD *)v2);
          return v8;
        }
      }
      else
      {
        v4 = (_QWORD *)*(a1 - 3);
        v2 = 0;
        v5 = 0;
      }
    }
    else
    {
      v4 = (_QWORD *)*(a1 - 3);
      v2 = 0;
      v3 = 0;
      v5 = 0;
    }
  }
  else
  {
    v2 = 0;
    v3 = 0;
    v4 = 0;
    v5 = 0;
  }
  v14 = (__int64 *)*(a1 - 3);
  v19 = 1;
  v17 = "exn.obj";
  v18 = 3;
  v16 = 0;
  v6 = sub_1648A60(88, 1u);
  v7 = a1;
  v8 = (__int64)v6;
  if ( v6 )
  {
    v9 = sub_15FB2A0(*v14, &v16, 1);
    sub_15F1EA0(v8, v9, 62, v8 - 24, 1, (__int64)a1);
    if ( *(_QWORD *)(v8 - 24) )
    {
      v10 = *(_QWORD *)(v8 - 16);
      v11 = *(_QWORD *)(v8 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *(_QWORD *)(v8 - 24) = v14;
    v12 = v14[1];
    *(_QWORD *)(v8 - 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (v8 - 16) | *(_QWORD *)(v12 + 16) & 3LL;
    *(_QWORD *)(v8 - 8) = (unsigned __int64)(v14 + 1) | *(_QWORD *)(v8 - 8) & 3LL;
    v14[1] = v8 - 24;
    *(_QWORD *)(v8 + 56) = v8 + 72;
    *(_QWORD *)(v8 + 64) = 0x400000000LL;
    sub_15FB110(v8, &v16, 1, (__int64)&v17);
    v7 = a1;
  }
  sub_15F20C0(v7);
  if ( v5 )
    goto LABEL_18;
  return v8;
}
