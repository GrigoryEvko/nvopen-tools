// Function: sub_33576B0
// Address: 0x33576b0
//
__int64 __fastcall sub_33576B0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6, __int64 a7)
{
  char *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // r15
  char *i; // r13
  __int64 v15; // rbx
  bool v16; // zf
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  _DWORD *v19; // r14
  char v20; // dl
  int v21; // ebx
  unsigned __int64 v22; // rdx
  unsigned int *v23; // r13
  char v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  char *v28; // [rsp+8h] [rbp-88h]
  const void *v29; // [rsp+10h] [rbp-80h]
  _QWORD *v30; // [rsp+18h] [rbp-78h]
  _QWORD *v31; // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+30h] [rbp-60h]
  unsigned __int16 *v34; // [rsp+48h] [rbp-48h]
  unsigned int v35[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v29 = (const void *)(a5 + 16);
  v9 = sub_E922F0(a6, a2);
  result = (__int64)&v9[2 * v11];
  v34 = (unsigned __int16 *)result;
  v30 = (_QWORD *)(a4 + 32);
  if ( v9 != (char *)result )
  {
    v13 = a4;
    for ( i = v9; v34 != (unsigned __int16 *)i; i += 2 )
    {
      v15 = *(unsigned __int16 *)i;
      result = *(_QWORD *)(a3 + 8 * v15);
      if ( result != a1 )
      {
        if ( result )
        {
          if ( !a7 || (result = *(_QWORD *)result, a7 != result) )
          {
            v16 = *(_QWORD *)(v13 + 72) == 0;
            v35[0] = *(unsigned __int16 *)i;
            if ( v16 )
            {
              v17 = *(_QWORD *)v13;
              v18 = *(unsigned int *)(v13 + 8);
              v19 = (_DWORD *)(*(_QWORD *)v13 + 4 * v18);
              if ( *(_DWORD **)v13 == v19 )
              {
                if ( v18 <= 3 )
                  goto LABEL_20;
              }
              else
              {
                result = *(_QWORD *)v13;
                while ( (_DWORD)v15 != *(_DWORD *)result )
                {
                  result += 4;
                  if ( v19 == (_DWORD *)result )
                    goto LABEL_19;
                }
                if ( v19 != (_DWORD *)result )
                  continue;
LABEL_19:
                if ( v18 <= 3 )
                {
LABEL_20:
                  v22 = v18 + 1;
                  if ( v22 > *(unsigned int *)(v13 + 12) )
                  {
                    sub_C8D5F0(v13, (const void *)(v13 + 16), v22, 4u, v10, v17);
                    v19 = (_DWORD *)(*(_QWORD *)v13 + 4LL * *(unsigned int *)(v13 + 8));
                  }
                  *v19 = v15;
                  ++*(_DWORD *)(v13 + 8);
                  goto LABEL_16;
                }
                v28 = i;
                v23 = *(unsigned int **)v13;
                v32 = *(_QWORD *)v13 + 4 * v18;
                do
                {
                  v26 = sub_B9AB10(v30, v13 + 40, v23);
                  if ( v27 )
                  {
                    v24 = v26 || v27 == v13 + 40 || *v23 < *(_DWORD *)(v27 + 32);
                    v31 = (_QWORD *)v27;
                    v25 = sub_22077B0(0x28u);
                    *(_DWORD *)(v25 + 32) = *v23;
                    sub_220F040(v24, v25, v31, (_QWORD *)(v13 + 40));
                    ++*(_QWORD *)(v13 + 72);
                  }
                  ++v23;
                }
                while ( (unsigned int *)v32 != v23 );
                i = v28;
              }
              *(_DWORD *)(v13 + 8) = 0;
              sub_B99770((__int64)v30, v35);
LABEL_16:
              v21 = *(unsigned __int16 *)i;
              result = *(unsigned int *)(a5 + 8);
              if ( result + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
              {
                sub_C8D5F0(a5, v29, result + 1, 4u, v10, v17);
                result = *(unsigned int *)(a5 + 8);
              }
              *(_DWORD *)(*(_QWORD *)a5 + 4 * result) = v21;
              ++*(_DWORD *)(a5 + 8);
              continue;
            }
            result = sub_B99770((__int64)v30, v35);
            if ( v20 )
              goto LABEL_16;
          }
        }
      }
    }
  }
  return result;
}
