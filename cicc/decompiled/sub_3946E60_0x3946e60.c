// Function: sub_3946E60
// Address: 0x3946e60
//
__int64 *__fastcall sub_3946E60(__int64 *a1)
{
  __int64 v1; // rbx
  void *v2; // r14
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // r12

  v1 = sub_22077B0(0x90u);
  if ( v1 )
  {
    memset((void *)v1, 0, 0x90u);
    v2 = (void *)(v1 + 112);
    *(_DWORD *)(v1 + 20) = 16;
    *(_QWORD *)(v1 + 64) = v1 + 112;
    *(_QWORD *)(v1 + 72) = 1;
    *(_DWORD *)(v1 + 96) = 1065353216;
    v3 = sub_222D860(v1 + 96, 0x100u);
    v5 = v3;
    if ( v3 > *(_QWORD *)(v1 + 72) )
    {
      if ( v3 == 1 )
      {
        *(_QWORD *)(v1 + 112) = 0;
      }
      else
      {
        if ( v3 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v1 + 96, 256, v4);
        v2 = (void *)sub_22077B0(8 * v3);
        memset(v2, 0, 8 * v5);
      }
      *(_QWORD *)(v1 + 64) = v2;
      *(_QWORD *)(v1 + 72) = v5;
    }
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
  }
  *a1 = v1;
  return a1;
}
