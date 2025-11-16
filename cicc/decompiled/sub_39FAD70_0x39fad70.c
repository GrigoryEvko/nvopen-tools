// Function: sub_39FAD70
// Address: 0x39fad70
//
int __fastcall sub_39FAD70(char *filename, struct stat *stat_buf)
{
  return __lxstat(1, filename, stat_buf);
}
