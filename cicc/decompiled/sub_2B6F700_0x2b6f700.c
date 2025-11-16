// Function: sub_2B6F700
// Address: 0x2b6f700
//
char __fastcall sub_2B6F700(__int64 *a1, __int64 a2, __int64 a3)
{
  char result; // al
  __int64 v6; // rdi
  __int64 v7; // r14
  unsigned __int8 v8; // al
  unsigned int v9; // r15d
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // r13d
  unsigned int v14; // r14d
  unsigned int v15; // r15d
  unsigned int v16; // eax
  unsigned int v17; // r11d
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // rbx
  unsigned int v21; // r15d
  __int64 v22; // r9
  unsigned int v23; // r14d
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int8 *v28; // r13
  unsigned __int8 *v29; // r12
  unsigned int v30; // edi
  unsigned int v31; // esi
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // eax
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // edx
  int v44; // eax
  __int64 v45; // [rsp-70h] [rbp-70h]
  __int64 v46; // [rsp-68h] [rbp-68h]
  unsigned int v47; // [rsp-60h] [rbp-60h]
  unsigned int v48; // [rsp-5Ch] [rbp-5Ch]
  __int64 *v49; // [rsp-58h] [rbp-58h]
  __int64 v50; // [rsp-50h] [rbp-50h]
  __int64 v51; // [rsp-50h] [rbp-50h]
  __int64 v52[9]; // [rsp-48h] [rbp-48h] BYREF

  result = 0;
  if ( a3 != a2 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
    v7 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
    v8 = *(_BYTE *)(v7 + 8);
    if ( *(_BYTE *)(v6 + 8) >= v8 )
    {
      if ( *(_BYTE *)(v6 + 8) > v8 )
        return 0;
      v9 = sub_BCB060(v6);
      v10 = sub_BCB060(v7);
      if ( v9 >= v10 )
      {
        if ( v9 > v10 )
          return 0;
        v11 = *a1;
        v12 = *(_QWORD *)(*a1 + 40);
        v13 = *(_WORD *)(a2 + 2) & 0x3F;
        v14 = *(_WORD *)(a3 + 2) & 0x3F;
        v50 = v12;
        v49 = *(__int64 **)(v11 + 16);
        v15 = sub_B52F50(v13);
        v16 = sub_B52F50(v14);
        v17 = v15;
        if ( v13 <= v15 )
          v17 = v13;
        if ( v14 <= v16 )
          v16 = v14;
        if ( v17 >= v16 )
        {
          if ( v17 <= v16 )
          {
            v18 = v50;
            v19 = v15;
            v20 = 0;
            v21 = v14;
            v22 = a3;
            v23 = v13;
            v24 = a2;
            while ( 1 )
            {
              v25 = (unsigned int)v20;
              v26 = (unsigned int)(1 - v20);
              if ( v23 > v19 )
                v25 = (unsigned int)(1 - v20);
              v27 = 32 * v25;
              if ( v21 == v17 )
                v26 = v20;
              v28 = *(unsigned __int8 **)(v24 + v27 - 64);
              v29 = *(unsigned __int8 **)(v22 + 32 * v26 - 64);
              if ( v28 != v29 )
              {
                v30 = *v28;
                v31 = *v29;
                if ( v30 < v31 )
                  return 1;
                if ( v30 > v31 )
                  return 0;
                if ( (unsigned __int8)v30 > 0x1Cu && (unsigned __int8)v31 > 0x1Cu )
                {
                  v32 = *((_QWORD *)v28 + 5);
                  if ( v32 )
                  {
                    v33 = (unsigned int)(*(_DWORD *)(v32 + 44) + 1);
                    v34 = *(_DWORD *)(v32 + 44) + 1;
                  }
                  else
                  {
                    v33 = 0;
                    v34 = 0;
                  }
                  v35 = *(_DWORD *)(v18 + 32);
                  v36 = 0;
                  if ( v34 < v35 )
                    v36 = *(_QWORD *)(*(_QWORD *)(v18 + 24) + 8 * v33);
                  v37 = *((_QWORD *)v29 + 5);
                  if ( v37 )
                  {
                    v38 = (unsigned int)(*(_DWORD *)(v37 + 44) + 1);
                    v39 = *(_DWORD *)(v37 + 44) + 1;
                  }
                  else
                  {
                    v38 = 0;
                    v39 = 0;
                  }
                  if ( v35 <= v39 )
                  {
                    if ( v36 )
                      return 0;
                    return v36 != 0;
                  }
                  v40 = *(_QWORD *)(*(_QWORD *)(v18 + 24) + 8 * v38);
                  if ( !v36 )
                  {
                    v36 = *(_QWORD *)(*(_QWORD *)(v18 + 24) + 8 * v38);
                    return v36 != 0;
                  }
                  if ( !v40 )
                    return 0;
                  if ( v40 != v36 )
                    return *(_DWORD *)(v36 + 72) < *(_DWORD *)(v40 + 72);
                  v48 = v19;
                  v45 = v22;
                  v46 = v24;
                  v47 = v17;
                  v51 = v18;
                  v52[0] = (__int64)v28;
                  v52[1] = (__int64)v29;
                  v41 = sub_2B5F980(v52, 2u, v49);
                  v18 = v51;
                  v19 = v48;
                  v17 = v47;
                  v24 = v46;
                  v22 = v45;
                  if ( v41 == 0 || v42 == 0 || v41 != v42 )
                  {
                    v43 = *v28;
                    v44 = *v29;
                    if ( (_BYTE)v43 != (_BYTE)v44 )
                      return v43 - 29 < (unsigned int)(v44 - 29);
                  }
                }
              }
              if ( (_DWORD)v20 == 1 )
                return 0;
              v20 = 1;
            }
          }
          return 0;
        }
      }
    }
    return 1;
  }
  return result;
}
