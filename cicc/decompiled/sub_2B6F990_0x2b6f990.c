// Function: sub_2B6F990
// Address: 0x2b6f990
//
__int64 __fastcall sub_2B6F990(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v7; // rdi
  int v9; // r15d
  __int64 *v10; // rax
  unsigned int v11; // r15d
  unsigned int v12; // r14d
  unsigned int v13; // eax
  unsigned int v14; // r8d
  unsigned int v15; // r9d
  __int64 i; // r10
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int8 v22; // si
  __int64 v23; // rdx
  int v24; // [rsp+Ch] [rbp-54h]
  __int64 *v25; // [rsp+10h] [rbp-50h]
  unsigned int v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+1Ch] [rbp-44h]
  unsigned int v28; // [rsp+1Ch] [rbp-44h]
  __int64 v29[8]; // [rsp+20h] [rbp-40h] BYREF

  LODWORD(v3) = 1;
  if ( a3 != a2 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
    v3 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
    if ( *(_BYTE *)(v7 + 8) != *(_BYTE *)(v3 + 8) )
      goto LABEL_3;
    v9 = sub_BCB060(v7);
    if ( v9 != (unsigned int)sub_BCB060(v3) )
      goto LABEL_3;
    v10 = *(__int64 **)(*(_QWORD *)a1 + 16LL);
    v11 = *(_WORD *)(a2 + 2) & 0x3F;
    v12 = *(_WORD *)(a3 + 2) & 0x3F;
    v25 = v10;
    v27 = sub_B52F50(v11);
    v13 = sub_B52F50(v12);
    v14 = v27;
    v15 = v27;
    if ( v11 <= v27 )
      v15 = v11;
    if ( v12 <= v13 )
      v13 = v12;
    if ( v13 == v15 )
    {
      for ( i = 0; ; i = 1 )
      {
        v17 = (unsigned int)i;
        v18 = (unsigned int)(1 - i);
        if ( v11 > v14 )
          v17 = (unsigned int)(1 - i);
        v19 = 32 * v17;
        if ( v12 == v15 )
          v18 = i;
        v20 = *(_QWORD *)(a2 + v19 - 64);
        v21 = *(_QWORD *)(a3 + 32 * v18 - 64);
        if ( v20 != v21 )
        {
          v22 = *(_BYTE *)v20;
          if ( *(_BYTE *)v20 != *(_BYTE *)v21 )
            goto LABEL_3;
          LOBYTE(v3) = v22 <= 0x1Cu;
          if ( v22 > 0x1Cu )
          {
            v24 = i;
            v26 = v15;
            v28 = v14;
            if ( *(_QWORD *)(v20 + 40) != *(_QWORD *)(v21 + 40) )
              break;
            v29[0] = v20;
            v29[1] = v21;
            if ( !sub_2B5F980(v29, 2u, v25) )
              break;
            v14 = v28;
            v15 = v26;
            LODWORD(i) = v24;
            if ( !v23 )
              break;
          }
        }
        if ( (_DWORD)i == 1 )
        {
          LODWORD(v3) = 1;
          return (unsigned int)v3;
        }
      }
    }
    else
    {
LABEL_3:
      LODWORD(v3) = 0;
    }
  }
  return (unsigned int)v3;
}
