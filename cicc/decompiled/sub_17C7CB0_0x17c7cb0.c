// Function: sub_17C7CB0
// Address: 0x17c7cb0
//
void __fastcall sub_17C7CB0(__int64 a1)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rcx
  _QWORD *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rdi
  double v8; // xmm0_8
  double v9; // xmm0_8
  unsigned __int64 v10; // r12
  _QWORD *v11; // r13
  __int64 *v12; // rax
  __int64 *v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r14
  _QWORD *v17; // rax
  const void *v18; // r12
  _BYTE *v19; // rsi
  _QWORD v20[2]; // [rsp-88h] [rbp-88h] BYREF
  _QWORD v21[4]; // [rsp-78h] [rbp-78h] BYREF
  const void *v22[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v23; // [rsp-48h] [rbp-48h] BYREF

  if ( byte_4FA3A60 )
  {
    if ( !sub_17C5230(*(_QWORD *)(a1 + 40)) )
    {
      if ( *(_DWORD *)(a1 + 128) )
      {
        v2 = *(_QWORD **)(a1 + 120);
        v3 = &v2[4 * *(unsigned int *)(a1 + 136)];
        if ( v2 != v3 )
        {
          while ( 1 )
          {
            v4 = v2;
            if ( *v2 != -16 && *v2 != -8 )
              break;
            v2 += 4;
            if ( v3 == v2 )
              return;
          }
          if ( v2 != v3 )
          {
            v5 = 0;
            do
            {
              v6 = *((unsigned int *)v4 + 2);
              v7 = *((unsigned int *)v4 + 3);
              v4 += 4;
              v5 += v7 + v6;
              if ( v4 == v3 )
                break;
              while ( *v4 == -8 || *v4 == -16 )
              {
                v4 += 4;
                if ( v3 == v4 )
                  goto LABEL_16;
              }
            }
            while ( v3 != v4 );
LABEL_16:
            if ( v5 )
            {
              if ( v5 < 0 )
                v8 = (double)(int)(v5 & 1 | ((unsigned __int64)v5 >> 1))
                   + (double)(int)(v5 & 1 | ((unsigned __int64)v5 >> 1));
              else
                v8 = (double)(int)v5;
              v9 = v8 * *(double *)&qword_4FA3980;
              if ( v9 >= 9.223372036854776e18 )
                v10 = (unsigned int)(int)(v9 - 9.223372036854776e18) ^ 0x8000000000000000LL;
              else
                v10 = (unsigned int)(int)v9;
              if ( v10 <= 9 )
              {
                LODWORD(v10) = 2 * v10;
                if ( (int)v10 < 10 )
                  LODWORD(v10) = 10;
                v10 = (int)v10;
              }
              v11 = **(_QWORD ***)(a1 + 40);
              v21[0] = sub_1643360(v11);
              v21[1] = sub_1643360(v11);
              v21[2] = sub_16471D0(v11, 0);
              v12 = (__int64 *)sub_1645600(v11, v21, 3, 0);
              v13 = sub_1645D80(v12, v10);
              v20[1] = 17;
              v16 = sub_15A06D0((__int64 **)v13, v10, v14, v15);
              v20[0] = "__llvm_prf_vnodes";
              LOWORD(v23) = 261;
              v22[0] = v20;
              v17 = sub_1648A60(88, 1u);
              v18 = v17;
              if ( v17 )
                sub_15E51E0((__int64)v17, *(_QWORD *)(a1 + 40), (__int64)v13, 0, 8, v16, (__int64)v22, 0, 0, 0, 0);
              sub_1694890((__int64)v22, 4, *(_DWORD *)(a1 + 100), 1u);
              sub_15E5D20((__int64)v18, v22[0], (size_t)v22[1]);
              if ( v22[0] != &v23 )
                j_j___libc_free_0(v22[0], v23 + 1);
              v22[0] = v18;
              v19 = *(_BYTE **)(a1 + 152);
              if ( v19 == *(_BYTE **)(a1 + 160) )
              {
                sub_167C6C0(a1 + 144, v19, v22);
              }
              else
              {
                if ( v19 )
                {
                  *(_QWORD *)v19 = v18;
                  v19 = *(_BYTE **)(a1 + 152);
                }
                *(_QWORD *)(a1 + 152) = v19 + 8;
              }
            }
          }
        }
      }
    }
  }
}
