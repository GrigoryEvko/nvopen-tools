// Function: sub_2FC9AB0
// Address: 0x2fc9ab0
//
__int64 __fastcall sub_2FC9AB0(__int64 a1, __int64 a2, unsigned int a3, _BYTE *a4, unsigned __int8 a5, char a6)
{
  char v6; // al
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r8
  unsigned int v16; // r15d
  _QWORD *v17; // rbx
  unsigned int v18; // eax
  _QWORD *v20; // [rsp+0h] [rbp-50h]
  char v22; // [rsp+Ch] [rbp-44h]
  unsigned int v23; // [rsp+Ch] [rbp-44h]
  _QWORD v24[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( !a1 )
    return 0;
  v6 = *(_BYTE *)(a1 + 8);
  if ( v6 != 16 )
  {
LABEL_11:
    if ( v6 == 15 )
    {
      v14 = *(_QWORD **)(a1 + 16);
      v20 = &v14[*(unsigned int *)(a1 + 12)];
      if ( v20 != v14 )
      {
        v15 = a5;
        v16 = 0;
        v17 = *(_QWORD **)(a1 + 16);
        while ( 1 )
        {
          v23 = v15;
          v18 = sub_2FC9AB0(*v17, a2, a3, a4, v15, 1);
          v15 = v23;
          if ( (_BYTE)v18 )
          {
            if ( *a4 )
              return 1;
            v16 = v18;
          }
          if ( v20 == ++v17 )
            return v16;
        }
      }
    }
    return 0;
  }
  if ( !sub_BCAC40(*(_QWORD *)(a1 + 24), 8) && !a5 )
  {
    if ( a6 )
      return 0;
    v10 = *(unsigned int *)(a2 + 276);
    if ( (unsigned int)v10 > 0x1F )
      return 0;
    v11 = 3623879202LL;
    if ( !_bittest64(&v11, v10) )
      return 0;
  }
  v22 = sub_AE5020(a2 + 312, a1);
  v12 = sub_9208B0(a2 + 312, a1);
  v24[1] = v13;
  v24[0] = ((1LL << v22) + ((unsigned __int64)(v12 + 7) >> 3) - 1) >> v22 << v22;
  if ( a3 <= (unsigned __int64)sub_CA1930(v24) )
  {
    *a4 = 1;
    return 1;
  }
  if ( !a5 )
  {
    v6 = *(_BYTE *)(a1 + 8);
    goto LABEL_11;
  }
  return 1;
}
