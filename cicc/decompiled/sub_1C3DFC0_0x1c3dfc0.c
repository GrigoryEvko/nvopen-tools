// Function: sub_1C3DFC0
// Address: 0x1c3dfc0
//
__int64 __fastcall sub_1C3DFC0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v4; // r12d
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 i; // r13
  __int64 v14; // r12
  __int64 *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // [rsp+10h] [rbp-60h]
  int v18; // [rsp+1Ch] [rbp-54h]
  char *v19; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  result = 0;
  if ( !LOBYTE(qword_4FBA380[20]) )
  {
    v20 = 257;
    if ( *off_4CD4988 )
    {
      v19 = off_4CD4988;
      LOBYTE(v20) = 3;
    }
    v2 = sub_1632310(a1, (__int64)&v19);
    v3 = v2;
    if ( v2 )
    {
      v4 = 0;
      v18 = sub_161F520(v2);
      if ( v18 )
      {
        do
        {
          v5 = sub_161F530(v3, v4);
          v6 = *(unsigned int *)(v5 + 8);
          v7 = *(_QWORD *)(v5 - 8 * v6);
          if ( v7 )
          {
            if ( *(_BYTE *)v7 == 1 )
            {
              v17 = *(_QWORD *)(v7 + 136);
              if ( !*(_BYTE *)(v17 + 16) && (unsigned int)v6 > 1 )
              {
                v8 = (unsigned int)v6;
                v9 = 1;
                while ( 1 )
                {
                  v10 = sub_161E970(*(_QWORD *)(v5 + 8 * (v9 - v8)));
                  if ( v11 == 6 && *(_DWORD *)v10 == 1852990827 && *(_WORD *)(v10 + 4) == 27749 )
                  {
                    v9 += 2;
                    sub_1C2EFB0(v17, 1);
                    if ( (unsigned int)v6 <= (unsigned int)v9 )
                      break;
                  }
                  else
                  {
                    v9 += 2;
                    if ( (unsigned int)v6 <= (unsigned int)v9 )
                      break;
                  }
                  v8 = *(unsigned int *)(v5 + 8);
                }
              }
            }
          }
          ++v4;
        }
        while ( v18 != v4 );
      }
      v12 = *(_QWORD *)(a1 + 32);
      for ( i = a1 + 24; i != v12; v12 = *(_QWORD *)(v12 + 8) )
      {
        while ( 1 )
        {
          v14 = v12 - 56;
          if ( !v12 )
            v14 = 0;
          if ( !sub_15602E0((_QWORD *)(v14 + 112), "nvvm.annotations_transplanted", 0x1Du) )
            break;
          v12 = *(_QWORD *)(v12 + 8);
          if ( i == v12 )
            return 1;
        }
        v15 = (__int64 *)sub_15E0530(v14);
        v16 = sub_155D020(v15, "nvvm.annotations_transplanted", 0x1Du, 0, 0);
        sub_15E0DA0(v14, -1, (__int64)v16);
      }
    }
    return 1;
  }
  return result;
}
